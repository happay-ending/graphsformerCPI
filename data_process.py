import gzip
import logging
import operator
import shutil
import threading
import time

import pandas as pd

from Bio import SeqIO
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.metrics import pairwise_distances

from biotoolbox.contact_map_builder import DistanceMapBuilder
from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from utils import *


#
#
# # mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    # print(allowable_set)
    # print("氨基酸的onehot编码:",list(map(lambda s: x == s, allowable_set)))
    # print(x)
    # print("allowable_set",allowable_set)
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    # mol = Chem.MolFromSmiles(smile)
    try:
        mol = MolFromSmiles(smile)
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            AllChem.Compute2DCoords(mol)

        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            print("atom:",atom.GetSymbol())
            feature = atom_features(atom)
            features.append(feature / sum(feature))
        node_features = np.array(features)

        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

        conf = mol.GetConformer()
        pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                               for k in range(mol.GetNumAtoms())])  # 原子的坐标

        # print("adj_matrix",adj_matrix)
        dist_matrix = pairwise_distances(pos_matrix)  # 原子距离矩阵
        # print("dist_matrix",dist_matrix)
        return c_size, node_features, adj_matrix, dist_matrix

    except ValueError as e:
        logging.warning(
            'the SMILES ({}) can not be converted to a graph, please check it before executing the program!\nREASON: {}'.format(
                smile, e))
        exit()




def target_to_cmap(target_key, target_sequence, contact_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    dist_matrix = np.load(contact_file)  # [氨基酸个数，氨基酸个数]

    contact_map = (dist_matrix < 8.0).astype(np.int32)  # 距离小于8A视为有接触

    # target_feature = target_to_feature(target_key, target_sequence, aln_dir)  #获取pid=target_key的对应的特征
    # print("1.target_feature shape:",target_feature.shape)
    # target_edge_index = np.array(target_edge_index)
    #################################################
    # 法2，利用get_residue_features获取特征向量
    from data_utils import get_residue_features
    feat = []
    for residue in target_sequence:
        residue_features = get_residue_features(residue)
        feat.append(residue_features)
    target_feature = torch.FloatTensor(feat)

    return target_size, target_feature, contact_map, dist_matrix


def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def make_distance_maps(pdbfile, chain=None, sequence=None):
    """
    Generate (diagonalized) C_alpha and C_beta distance matrix from a pdbfile
    """
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    # structure_container.chains = {chain: structure_container.chains[chain]}

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()

    return ca.chains, cb.chains


def load_predicted_PDB(pdbfile):
    '''备用代码，读pdb文件中的残基距离矩阵和残基序列，注意：不考虑chain类型, C_alpha distance matrix。'''
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)

    return distances, seqs[0]


def load_FASTA(filename):
    '''读fasta文件，返回proteins_id和sequenc两个列表'''
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'r')
    entries = []
    proteins_id = []
    for entry in SeqIO.parse(infile, 'fasta'):
        entries.append(str(entry.seq))
        proteins_id.append(str(entry.id))
    return proteins_id, entries


def un_gz(file_name, new_file_name):
    '''解压gz文件到new_file_name文件'''
    # 开始解压
    g_file = gzip.GzipFile(file_name)
    # 读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
    open(new_file_name, "wb+").write(g_file.read())
    g_file.close()


def get_prt(file_path, save_path):
    '''读取human、celegens数据集中蛋白质序列并编码保存成CSV文件'''
    human_data = pd.read_csv(file_path, sep=' ', header=None)
    human_data.columns = ['smiles', 'seq', 'label']
    prt = human_data['seq'].drop_duplicates()
    prt.reset_index(inplace=True, drop=True)
    print(len(prt))
    h_id = []
    for i in range(len(prt)):
        h_id.append("h_" + str(i))

    h_id = pd.Series(h_id, name='h_id')
    prt = pd.concat([h_id, prt], axis=1)
    prt.to_csv(save_path, index=False)


from concurrent.futures import ThreadPoolExecutor, as_completed, wait

mutex = threading.Lock()


def query_fasta_with_threads(*args):
    index, key, seq, fasta_file, saved_file = args
    time.sleep(1)
    thread_name = str(index) + threading.current_thread().name
    prt_seq_len = len(str(seq).strip())

    print('当前线程:{}在fasta文件中搜索第{}个蛋白质!'.format(thread_name, index))
    infile = open(fasta_file, 'r')
    for entry in SeqIO.parse(infile, 'fasta'):
        uniprot_prt_len = len(str(entry.seq).strip())
        if prt_seq_len == uniprot_prt_len:
            if operator.eq(str(entry.seq).strip(), str(seq).strip()):
                uniprot_id = entry.id.split(sep='|')[1]
                # if uniprot_id.startswith('O') or uniprot_id.startswith('P') or uniprot_id.startswith('Q'):
                entries = [str(index), key, uniprot_id]
                print('\t'.join(entries))
                mutex.acquire()
                with open(saved_file, mode='a', encoding="utf-8") as f:
                    f.write('\t'.join(entries) + '\n')
                mutex.release()
                return key + " 成功找到!"
    entries = [str(index), key, "0"]
    mutex.acquire()
    with open(saved_file, mode='a', encoding="utf-8") as f:
        f.write('\t'.join(entries) + '\n')
    mutex.release()
    return key + " 没有找到！"


def get_id_map(prt_file, fasta_file, saved_map_file):
    '''1.多线程读取蛋白质序列，并在指定的fasta文件中读取对应的uniprot ID，返回id映射文件。。'''
    davis_prt_dict = json.load(open(prt_file), object_pairs_hook=OrderedDict)
    start_time = time.time()
    future_list = []
    executor = ThreadPoolExecutor(max_workers=5)  # 创建 ThreadPoolExecutor
    for i, item in enumerate(davis_prt_dict):
        task = executor.submit(query_fasta_with_threads, *(i, item, davis_prt_dict[item], fasta_file, saved_map_file))
        future_list.append(task)  # 提交任务

    # 以下备注用于human 或 celegan数据集
    # prt_file = pd.read_csv(prt_file)
    # start_time = time.time()
    # future_list = []
    # executor = ThreadPoolExecutor(max_workers=5)  # 创建 ThreadPoolExecutor
    # for i, item in enumerate(prt_file.values):
    #     task = executor.submit(query_fasta_with_threads, *(i, item[0], item[1], fasta_file, saved_map_file))
    #     future_list.append(task)  # 提交任务

    wait(future_list)

    for future in as_completed(future_list):
        result = future.result()  # 获取任务结果
        print("%s get result : %s" % (threading.current_thread().getName(), result))

    print('%s cost %d second' % (threading.current_thread().getName(), time.time() - start_time))


def get_alphafold_pdb():
    '''在Alphafold数据集中查询蛋白质uniprot id对应的pdb文件，并存入pdb文件夹，同时统计没有对应pdb文件的蛋白质'''
    dataset = 'davis'  # kiba,davis,human,celegans

    pdb_dir = os.path.join('data', dataset, 'pdb')
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    protein_path = os.path.join('data', dataset)
    if dataset == 'kiba':
        proteins = json.load(open(os.path.join(protein_path, 'proteins.txt')), object_pairs_hook=OrderedDict)
        print("proteins", len(proteins))
        uniprot_ids = list(proteins.keys())
    elif dataset == 'human' or dataset =='celegans' or dataset =='davis':
        id_map = pd.read_csv(os.path.join(protein_path, 'uniprot_id_map.txt'), sep='\t', header=0)
        uniprot_ids = id_map['uniprot_id']

    # alphafold压缩文件的位置，AF文件夹中存放alphafold总文件
    # alphafold_path = 'D:\\AF'
    # alphafold_path = 'E:\\AF-celegans'
    alphafold_path = 'E:\\AF-human'
    alphafold_pdbs = os.listdir(alphafold_path)
    pdbs = []
    for pdb in alphafold_pdbs:
        pdbs.append(pdb.split('-')[1])

    flag = [0] * len(uniprot_ids)
    for i, key in enumerate(uniprot_ids):
        print("-" * 50)
        print("第{}个蛋白质：{}".format(i, key))
        if key in pdbs:
            print("该压缩pdb文件存在")
            source_path = os.path.join(alphafold_path, 'AF-' + key + '-F1-model_v4.pdb.gz')
            print(source_path)
            target_path = os.path.join(pdb_dir, key + ".pdb")
            print(target_path)
            un_gz(source_path, target_path)
            flag[i] = 1
    flag = pd.Series(flag, name='flag')
    df_s = pd.concat([id_map, flag], axis=1)
    df_s.to_csv(os.path.join(protein_path, 'no_pdb.csv'), index=False)


def get_alphafold_cmap():
    '''根据pdb文件生成蛋白质距离矩阵'''
    dataset = 'davis'  # kiba,davis,human,celegans

    pdb_dir = os.path.join('data', dataset, 'pdb')
    cmap_dir = os.path.join('data', dataset, 'cmap')
    if not os.path.exists(cmap_dir):
        os.makedirs(cmap_dir)

    if dataset == 'kiba':
        prts = json.load(open(os.path.join('data', dataset, 'proteins.txt')),
                         object_pairs_hook=OrderedDict)  # nem*[pid,sequene]

        protein_dict = {}
        for key in prts.keys():
            pdb_path = os.path.join(pdb_dir, key + '.pdb')
            if os.path.exists(pdb_path):
                _, beta_c = make_distance_maps(pdb_path, chain='A')
                protein_dict[key] = beta_c['A']['seq']
                np.save(os.path.join(cmap_dir, key + '.npy'), beta_c['A']['contact-map'])
            else:
                protein_dict[key] = prts[key]
        with open(os.path.join('data', dataset, 'proteins(revised).txt'), 'w') as file:
            file.write(json.dumps(protein_dict))
    elif dataset =='human' or dataset =='celegans':
        prt = pd.read_csv(os.path.join('data', dataset, 'prt_pdb.csv'), header=0)
        seq = []
        for i, uniprot_id in enumerate(prt['uniprot_id']):
            print(i)
            print("uniprot_id:", uniprot_id)
            pdb_path = os.path.join(pdb_dir, uniprot_id + '.pdb')
            _, beta_c = make_distance_maps(pdb_path, chain='A')
            print(pdb_path)
            # cmap,sequence = load_predicted_PDB(pdb_path)
            # print('contact-map\n',alpha_c['A']['contact-map'])
            cmap = beta_c['A']['contact-map']
            sequence = beta_c['A']['seq']
            print('cmap', cmap)
            print('sequence\n', sequence)
            print(len(sequence))
            np.save(os.path.join(cmap_dir, uniprot_id + '.npy'), cmap)
            seq.append(sequence)
        seq = pd.Series(seq, name='seq')
        df_s = pd.concat([prt, seq], axis=1)
        df_s.to_csv(os.path.join('data', dataset, dataset + '_prt(uniprot_seq).csv'), index=False)
    elif dataset == 'davis':
        prt = pd.read_csv(os.path.join('data', dataset, 'uniprot_id_map.txt'), sep='\t', header=0)
        id_map_dict = {}
        for record in prt.values:
            id_map_dict[record[1]] = record[2]

        prts = json.load(open(os.path.join('data', dataset, 'proteins.txt')),
                         object_pairs_hook=OrderedDict)  # nem*[pid,sequene]

        protein_dict = {}
        for key in prts.keys():
            if key in id_map_dict.keys():
                pdb_path = os.path.join(pdb_dir, id_map_dict[key] + '.pdb')
                print(pdb_path)
                _, beta_c = make_distance_maps(pdb_path, chain='A')
                cmap = beta_c['A']['contact-map']
                sequence = beta_c['A']['seq']
                print('cmap', cmap)
                print('sequence\n', sequence)
                print("蛋白质长度：", len(sequence))
                np.save(os.path.join(cmap_dir, key + '.npy'), cmap)
                protein_dict[key] = sequence
            else:
                protein_dict[key] = prts[key]
        print("protein_dict", protein_dict)
        with open(os.path.join('data', dataset, 'proteins(revised).txt'), 'w') as file:
            file.write(json.dumps(protein_dict))


def rename_file():
    path = 'E:\AF-davis-searched'
    file_list = os.listdir(path)
    for i, fi in enumerate(file_list):
        old_file_name = os.path.join(path, fi)
        print('old_file_name', old_file_name)
        new_file_name = os.path.join(path, fi.split('-')[1] + '.pdb')
        print('new_file_name', new_file_name)
        try:
            os.rename(old_file_name, new_file_name)
        except Exception as e:
            print(e)
            print("Failed!")
        else:
            print("SUcess!")


def get_affiniy_class():
    '''得到化合物-蛋白质分类数据总集，数据中剔除了没有pdb文件的蛋白质,used for human and celegans'''
    root_path = 'data'
    dataset = 'human'  # human,celegans
    original_data = pd.read_csv(os.path.join(root_path, dataset, dataset + '.txt'), sep=' ', header=None)
    original_data.columns = ['smiles', 'seq', 'label']
    prt_old = pd.read_csv(os.path.join(root_path, dataset, dataset + '_prt.csv'), header=0)

    merge_t1 = pd.merge(left=original_data, right=prt_old, how="left", on='seq')
    merge_t1.drop('seq', axis=1)
    merge_t1 = merge_t1[['h_id', 'smiles', 'label']]
    print('merge_t1\n', merge_t1)

    prt_new = pd.read_csv(os.path.join(root_path, dataset, dataset + '_prt(uniprot_seq).csv'), header=0)
    merge_t2 = pd.merge(left=prt_new, right=merge_t1, how="inner", on='h_id')
    merge_t2 = merge_t2[['h_id', 'uniprot_id', 'smiles', 'seq', 'label']]
    merge_t2 = merge_t2.sample(frac=1.0)
    print('merge_t2\n', merge_t2)
    merge_t2.to_csv(os.path.join(root_path, dataset, dataset + '_data.csv'), index=False)

def move_file(srcfile, dstpath):  # 移动函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, dstpath + fname)  # 移动文件
        # print("move %s -> %s" % (srcfile, dstpath + fname))

def save_cpds_graph():
    root_path = 'data'
    dataset = 'kiba'  # human,celegans,davis,kiba
    cpds_graph_dir = os.path.join(root_path, dataset, 'cpd_graph')
    if not os.path.exists(cpds_graph_dir):
        os.makedirs(cpds_graph_dir)
    if dataset == 'human' or dataset == 'celegans':
        human_data = pd.read_csv(os.path.join(root_path, dataset + '_data.csv'), header=0)
        print(human_data)

        # 1.smile清洗处理
        human_data_no_smiles = human_data.drop(columns='smiles')
        smiles = []
        for i in human_data.smiles.tolist():
            cpd = str(i).split('.')
            cpd_longest = max(cpd, key=len)
            smiles.append(cpd_longest)

        smiles = pd.Series(smiles, name='smiles')
        human_data_cleaned = pd.concat([human_data_no_smiles, smiles], axis=1)
        smiles_unique, counts = np.unique(human_data_cleaned['smiles'], return_counts=True)
        print(
            "Unique smile : {}, Total number: {}, Statistical information :{}".format(smiles_unique, len(smiles_unique),
                                                                                      counts))
        cpd_id = []
        ###，2.去除重复的原子，
        for i, smile in enumerate(smiles_unique):
            print("第{}个化合物,它的smile格式是{}".format(i, smile))
            g = smile_to_graph(smile)  # g: c_size,node_features, adj_matrix, dist_matrix
            cpd_id.append("cid_" + str(i))
            np.savez(os.path.join(cpds_graph_dir, "cid_" + str(i)), atom_features=g[1], adj_matrix=g[2],
                     dist_matrix=g[3])

        df = pd.DataFrame({"cid": cpd_id, "smiles": smiles_unique})
        df.to_csv(os.path.join(root_path, dataset + '_cpd.csv'), index=False)

        human_data_cid = pd.merge(left=human_data_cleaned, right=df, how='inner', on='smiles')
        human_data_cid = human_data_cid.sample(frac=1.0)
        human_data_cid.to_csv(os.path.join(root_path, dataset + '_data(6).csv'), index=False)
    elif dataset == 'davis' or dataset =='kiba':
        print("davis or kiba")
        cpds = json.load(open(os.path.join(root_path,dataset, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
        cpd_keys = list(cpds.keys())
        for i, key in enumerate(cpd_keys):
            print("第{}个化合物,它的编号是{}".format(i, key))
            g = smile_to_graph(cpds[key])
            np.savez(os.path.join(cpds_graph_dir, key), atom_features=g[1], adj_matrix=g[2],
                     dist_matrix=g[3])
        print("finished !")


def split_regression_data(root: str, fold=0, rank=0, cpd_threshold=1.0, prt_threshold=1.0):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    sample_data = json.load(open(os.path.join(root, 'folds/train_fold_setting1.txt')))
    sample_data = [e for e in sample_data]  # train_fold_origin由5组数据组成

    train_folds = []
    # fold: 验证， fold! :训练
    valid_fold = sample_data[fold]  # one fold
    for i in range(len(sample_data)):  # other folds
        if i != fold:
            train_folds += sample_data[i]

    cpd_len_list = []
    cpds_graph_dir = os.path.join(root, 'cpd_graph')
    for each in os.listdir(cpds_graph_dir):
        npzfile = np.load(os.path.join(cpds_graph_dir, each))
        cpd_len_list.append(npzfile['atom_features'].shape[0])

    sort_cpd_len_list = np.sort(cpd_len_list)
    num = math.ceil(len(cpd_len_list) * cpd_threshold)
    cpd_max_len = 0
    for i, each in enumerate(sort_cpd_len_list):
        if i + 1 >= num and i < num:
            cpd_max_len = each
            break
    print("截断化合物分子最大长度：", cpd_max_len)

    prts = json.load(open(os.path.join(root, 'proteins(revised).txt')), object_pairs_hook=OrderedDict)
    sequences = np.array(list(prts.values()))
    prt_len_list = []
    for each in sequences:
        prt_len_list.append(len(each))
    sort_prt_len_list = np.sort(prt_len_list)
    num = math.ceil(len(prt_len_list) * prt_threshold)
    prt_max_len = 0
    for i, each in enumerate(sort_prt_len_list):
        if i + 1 >= num and i < num:
            prt_max_len = each
            break
    print("阈值对应蛋白质的长度", prt_max_len)

    contact_dir = os.path.join(root, 'cmap')
    contact_file_names = set()
    for each in os.listdir(contact_dir):
        contact_file_names.add(each.split(".npy")[0])
    missed_prt_file = list(set(prts.keys()) - contact_file_names)
    print("缺失的蛋白质文件是：", missed_prt_file)

    missed_prot_f_loc = []
    for f in missed_prt_file:
        missed_prot_f_loc.append(list(prts.keys()).index(f))
    print("缺失的contact文件在蛋白质字典中的位置是：", missed_prot_f_loc)

    # prt_graphs = {}
    # for prt_id in prts.keys():
    #     if prt_id not in missed_prt_file:
    #         g = target_to_cmap(prt_id, prts[prt_id],
    #                            contact_dir)  # target_size, target_feature, contact_map, dist_matrix
    #         prt_graphs[prt_id] = g
    #
    # prt_loc_2_id = {iloc: key for iloc, key in enumerate(list(prts.keys()))}
    #
    affinity = pickle.load(open(os.path.join(root, 'Y'), 'rb'), encoding='latin1')
    if 'davis' in root:
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    rows, cols = np.where(np.isnan(affinity) == False)
    # 有效的训练数据和有效的测试数据编号
    effective_train_folds = [x for x in train_folds if cols[x] not in missed_prot_f_loc]
    effective_valid_fold = [x for x in valid_fold if cols[x] not in missed_prot_f_loc]

    if rank == 0:
        print("-" * 57)
        print(
            "| Train entries:{}\t, effective train entries:{}\t|\n| Valid entries:{}\t, effective valid entries:{}\t|".format(
                len(train_folds), len(effective_train_folds), len(valid_fold), len(effective_valid_fold)))
        print("-" * 57)
        print('| Effective compounds:{}, effective proteins:{}\t|'.format(len(cpd_len_list), len(contact_file_names)))
        print("-" * 57)
    return effective_train_folds, effective_valid_fold, cpd_max_len, prt_max_len


# bak def split_regression_data(root: str, fold=0, rank=0, cpd_threshold=1.0, prt_threshold=1.0):
#     random.seed(0)  # 保证随机结果可复现
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
#
#     sample_data = json.load(open(os.path.join(root, 'folds/train_fold_setting1.txt')))
#     sample_data = [e for e in sample_data]  # train_fold_origin由5组数据组成
#
#     # load train,valid and test entries
#     train_folds = []
#     # fold: 验证， fold! :训练
#     valid_fold = sample_data[fold]  # one fold
#     for i in range(len(sample_data)):  # other folds
#         if i != fold:
#             train_folds += sample_data[i]
#
#     cpds = json.load(open(os.path.join(root, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
#     print("配体文件中化合物总数：", len(cpds))
#     # kiba数据集中有重复的smile值
#     cpd_len_list = []
#     cpd_graphs = {}
#     for key in cpds.keys():
#         g = smile_to_graph(cpds[key])  # g: c_size,node_features, adj_matrix, dist_matrix
#         cpd_graphs[key] = g
#         cpd_len_list.append(g[0])
#     sort_cpd_len_list = np.sort(cpd_len_list)
#     print(sort_cpd_len_list)
#     num = math.ceil(len(cpds) * cpd_threshold)
#     cpd_max_len = 0
#     for i, each in enumerate(sort_cpd_len_list):
#         if i + 1 >= num and i < num:
#             cpd_max_len = each
#             print("位置：", i)
#             break
#     print("截断化合物分子最大长度：", cpd_max_len)
#
#     prts = json.load(open(os.path.join(root, 'proteins.txt')), object_pairs_hook=OrderedDict)
#     sequences = np.array(list(prts.values()))
#     prt_len_list = []
#     for each in sequences:
#         prt_len_list.append(len(each))
#     sort_prt_len_list = np.sort(prt_len_list)
#     # print("蛋白质长度：", sort_prt_len_list)
#     # print("蛋白质个数：", len(len_list))
#     num = math.ceil(len(prt_len_list) * prt_threshold)
#     # print("阈值对应的蛋白质个数",num)
#     prt_max_len = 0
#     for i, each in enumerate(sort_prt_len_list):
#         if i + 1 >= num and i < num:
#             prt_max_len = each
#             break
#     print("阈值对应蛋白质的长度", prt_max_len)
#
#     contact_dir = os.path.join(root, 'cmap')
#     contact_file_names = set()
#     for each in os.listdir(contact_dir):
#         contact_file_names.add(each.split(".npy")[0])
#     missed_prt_file = list(set(prts.keys()) - contact_file_names)
#     print("缺失的蛋白质文件是：", missed_prt_file)
#
#     missed_prot_f_loc = []
#     for f in missed_prt_file:
#         missed_prot_f_loc.append(list(prts.keys()).index(f))
#     print("缺失的contact文件在蛋白质字典中的位置是：", missed_prot_f_loc)
#
#     prt_graphs = {}
#     for prt_id in prts.keys():
#         if prt_id not in missed_prt_file:
#             g = target_to_cmap(prt_id, prts[prt_id],
#                                contact_dir)  # target_size, target_feature, contact_map, dist_matrix
#             prt_graphs[prt_id] = g
#
#     prt_loc_2_id = {iloc: key for iloc, key in enumerate(list(prts.keys()))}
#
#     affinity = pickle.load(open(os.path.join(root, 'Y'), 'rb'), encoding='latin1')
#     if 'davis' in root:
#         affinity = [-np.log10(y / 1e9) for y in affinity]
#     affinity = np.asarray(affinity)
#     rows, cols = np.where(np.isnan(affinity) == False)
#     # 有效的训练数据和有效的测试数据编号
#     effective_train_folds = [x for x in train_folds if cols[x] not in missed_prot_f_loc]
#     effective_valid_fold = [x for x in valid_fold if cols[x] not in missed_prot_f_loc]
#
#     if rank == 0:
#         print("-" * 57)
#         print(
#             "| Train entries:{}\t, effective train entries:{}\t|\n| Valid entries:{}\t, effective valid entries:{}\t|".format(
#                 len(train_folds), len(effective_train_folds), len(valid_fold), len(effective_valid_fold)))
#         print("-" * 57)
#         print('| Effective compounds:{}, effective proteins:{}\t|'.format(len(cpd_graphs), len(prt_graphs)))
#         print("-" * 57)
#     return effective_train_folds, effective_valid_fold, cpd_max_len, cpd_graphs, prt_max_len, prt_graphs, prt_loc_2_id, affinity


def split_classified_data(root: str, rank=0, train_sample_ratio=0.8, cpd_threshold=1.0, prt_threshold=1.0):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    dataset = root.split('/')[-1]

    cpd_len_list = []
    cpds_graph_dir = os.path.join(root, 'cpd_graph')
    for each in os.listdir(cpds_graph_dir):
        npzfile = np.load(os.path.join(cpds_graph_dir, each))
        cpd_len_list.append(npzfile['atom_features'].shape[0])

    sort_cpd_len_list = np.sort(cpd_len_list)
    num = math.ceil(len(cpd_len_list) * cpd_threshold)
    cpd_max_len = 0
    for i, each in enumerate(sort_cpd_len_list):
        if i + 1 >= num and i < num:
            cpd_max_len = each
            break
    # print("截断化合物分子最大长度：", cpd_max_len)

    prt_data = pd.read_csv(os.path.join(root, dataset + '_prt(uniprot_seq).csv'), header=0)
    prt_len_list = []
    for each in prt_data['seq']:
        prt_len_list.append(len(each))
    sort_prt_len_list = np.sort(prt_len_list)
    num = math.ceil(len(prt_len_list) * prt_threshold)
    # print("阈值对应的蛋白质个数",num)
    prt_max_len = 0
    for i, each in enumerate(sort_prt_len_list):
        if i + 1 >= num and i < num:
            prt_max_len = each
            # print("蛋白质位置：", i)
            break
    # print("阈值对应蛋白质的长度", prt_max_len)

    human_data = pd.read_csv(os.path.join(root, dataset + '_data(6).csv'), header=0)
    # 生成训练样本和测试样本的索引
    random.seed(0)  # 保证随机结果可复现
    n = len(human_data)
    x = int(n * train_sample_ratio)
    train_index = random.sample(range(n), x)
    test_index = np.delete(np.arange(n), train_index)
    if rank == 0:
        print("-" * 53)
        print("| Effective train entries:{}\t\t\t\t\t\t|\n| Effective valid entries:{}\t\t\t\t\t\t|".format(
            len(train_index), len(test_index)))
        print("-" * 53)
        print('| Effective compounds:{}, effective proteins:{}\t|'.format(len(cpd_len_list), len(prt_len_list)))
        print("-" * 53)

    return train_index, test_index, cpd_max_len, prt_max_len


if __name__ == '__main__':
    # data_root_path = "C:\\Users\\Austin\\Downloads\\5luq.pdb"
    # data_root_path = "./data/kiba/pdb/O00141.pdb"
    # data_root_path= "F:/AF-Q00975-F1-model_v4.pdb"
    # # pdb_id = 'AF-Q5VSL9-F1-model_v4'
    # # sequence = 'MEPAVGGPGPLIVNNKQPQPPPPPPPAAAQPPPGAPRAAAGLLPGGKAREFNRNQRKDSEGYSESPDLEFEYADTDKWAAELSELYSYTEGPEFLMNRKCFEEDFRIHVTDKKWTELDTNQHRTHAMRLLDGLEVTAREKRLKVARAILYVAQGTFGECSSEAEVQSWMRYNIFLLLEVGTFNALVELLNMEIDNSAACSSAVRKPAISLADSTDLRVLLNIMYLIVETVHQECEGDKAEWRTMRQTFRAELGSPLYNNEPFAIMLFGMVTKFCSGHAPHFPMKKVLLLLWKTVLCTLGGFEELQSMKAEKRSILGLPPLPEDSIKVIRNMRAASPPASASDLIEQQQKRGRREHKALIKQDNLDAFNERDPYKADDSREEEEENDDDNSLEGETFPLERDEVMPPPLQHPQTDRLTCPKGLPWAPKVREKDIEMFLESSRSKFIGYTLGSDTNTVVGLPRPIHESIKTLKQHKYTSIAEVQAQMEEEYLRSPLSGGEEEVEQVPAETLYQGLLPSLPQYMIALLKILLAAAPTSKAKTDSINILADVLPEEMPTTVLQSMKLGVDVNRHKEVIVKAISAVLLLLLKHFKLNHVYQFEYMAQHLVFANCIPLILKFFNQNIMSYITAKNSISVLDYPHCVVHELPELTAESLEAGDSNQFCWRNLFSCINLLRILNKLTKWKHSRTMMLVVFKSAPILKRALKVKQAMMQLYVLKLLKVQTKYLGRQWRKSNMKTMSAIYQKVRHRLNDDWAYGNDLDARPWDFQAEECALRANIERFNARRYDRAHSNPDFLPVDNCLQSVLGQRVDLPEDFQMNYDLWLEREVFSKPISWEELLQ'
    # a, b = make_distance_maps(data_root_path, chain='A')
    # cmap = a['A']['contact-map']
    # print("a-cmap", cmap)
    # print('a-seq\n',a['A']['seq'])
    # print(type(cmap))
    # print("b", b['A']['contact-map'])
    # fasta_file = './data/davis/seq/AAK1.fasta'
    # fasta_file = 'C:\\Users\\Austin\\Downloads\\zz.fasta'
    # p,e = load_FASTA(fasta_file)
    # print(p)
    # print(e)

    # fasta_dir = './data/davis/seq/'
    # dirs = os.listdir(fasta_dir)
    # d = dict()
    # for file in dirs:
    #     file_path = fasta_dir+file
    #     proteins_id, seq = load_FASTA(file_path)
    #
    #     if seq[0] in d:
    #         d[seq[0]].append(proteins_id[0])
    #     else:
    #         d[seq[0]] = [proteins_id[0]]
    #
    # with open('./data/davis/proteins_dict.csv', 'w') as f:
    #     for key in d.keys():
    #         ids = ','.join(d[key])
    #         print(key, d[key],len(d[key]),ids)
    #         f.write("%s,%s\n" % (key, ids))

    # 1.在alphafold2文件夹中查找蛋白质pdb文件到c_map文件夹中
    # get_alphafold_pdb()
    # rename_file()
    # get_alphafold_cmap()
    # 保存化合物的graph
    # save_cpds_graph()
    # target_size, target_feature, contact_map, dist_matrix = target_to_cmap("A1IMB8", "A", "./data/celegans/cmap")
    # print(target_size)
    # print(contact_map)
    # print(dist_matrix)
    # c_size, node_features, adj_matrix, dist_matrix = smile_to_graph('CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1')
    # print(c_size)
    # print(adj_matrix)
    # print(dist_matrix)
    # get_affiniy_class()
    smile_to_graph('CC(=O)NC1=NN=C(S1)S(=O)(=O)N')