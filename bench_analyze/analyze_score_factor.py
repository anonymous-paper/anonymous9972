import argparse
import math

import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_analyze.criteria import criteria


def parse_cmd_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--savedir', type=str, default='./bench_analyze/results')
    parser.add_argument('--score_path', type=str, default='./search_space/results/vcnn_cifar100_seed777_dlrandom_dlinfo1_7168.p')
    parser.add_argument('--search_space_config', type=str, default='./configs/search_space/cifar100_SuperResConvKXKX.yaml')

    parser.add_argument('--factor', type=str, default=None)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def return_value(x):
    if x == []: return 1
    if type(x) != list:
        x = x
    elif len(x) > 2:
        # x = x[0][2]
        x = x[-1]
    else:
        x = x[0][-1]
    # x = x[0][2] if type(x) == list else x
    return x


def draw_sub(xs, ys, row, column, subtitle, title, path, xlabel, ylabel):
    # row, column = 2, 3
    plt.figure(figsize=(18,10))
    plt.suptitle(title, fontsize=16)
    # figure, ax = plt.subplots(row, column)
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.subplot(row, column, i+1)
        plt.ylabel('Actual Acc(%)', fontsize=12)
        plt.xlabel('Estimated Score', fontsize=12)
        t = 'first layer' if i == 0 else f'stage {i}'
        plt.title(t, fontsize=12) # title

        plt.scatter(x, y, marker='o')

    plt.savefig(os.path.join(path, title)+f'.{format}', bbox_inches='tight')
    plt.grid()
    plt.close()
    
    return


def scatter_cost_acc(accs, flops, path, title, marker='o', format='jpg'):
    accs = np.array(accs) * 100
    plt.scatter(flops, accs, marker=marker)
    
    plt.ylabel('Test Acc(%)', fontsize=12)
    plt.xlabel('FLOPs', fontsize=12)
    # plt.title(title, fontsize=16)
    plt.savefig(os.path.join(path, title)+f'.{format}', bbox_inches='tight')
    plt.savefig(os.path.join(path, title)+f'.jpg', bbox_inches='tight', dpi=600)

    plt.grid()
    plt.close()


def draw_scatter(arches, accs, measures, path, title, marker='o', format='jpg'):
    # row, column = 3, 4
    if len(measures) <= 13:
        # row, column = 3, 3
        # row, column = 3, 4
        row, column = 4, 3
        # plt.figure(figsize=(28,18))
        plt.figure(figsize=(22,28))
    else:
        row, column = 8, 10
        plt.figure(figsize=(60,50))
    # plt.suptitle(title, fontsize=16)
    # figure, ax = plt.subplots(row, column)
    for i, (k, v) in enumerate(measures.items()):
        # if k == 'Oracle': continue
        # if k in ['FLOPs', 'Params', 'Oracle']: continue
        # if k in ['FLOPs', 'Params', 'ntk', 'Oracle']: continue
        if k in ['ntk', 'Oracle']: continue
        plt.subplot(row, column, i+1)
        plt.ylabel('Actual Acc(%)', fontsize=12)
        plt.xlabel('Estimated Score', fontsize=12)
        # t = 'first layer' if i == 0 else f'stage {i}'
        plt.title(k, fontsize=12) # title

        plt.scatter(v, accs, marker='o')
    
    # plt.savefig(os.path.join(path, title)+f'.{format}', bbox_inches='tight')
    plt.savefig(os.path.join(path, title)+f'.jpg', bbox_inches='tight', dpi=600)
    plt.grid()
    plt.close()


def draw_scatter_individually(arches, accs, scores, title, path, marker='o', format='jpg'):
    # plt.subplot(row, column, i+1)
    plt.ylabel('Actual Acc(%)', fontsize=18)
    plt.xlabel('Estimated Score', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # t = 'first layer' if i == 0 else f'stage {i}'
    plt.title(title, fontsize=22) # title

    plt.scatter(scores, accs, marker='o')
    
    # plt.savefig(os.path.join(path, title)+f'.{format}', bbox_inches='tight')
    # add = '_201'
    add = '_zero'

    plt.savefig(os.path.join(path, title+add)+f'.jpg', bbox_inches='tight', dpi=200)
    plt.grid()
    plt.close()


def draw_params_flops_depth(arches, accs, measures, path, factor_type='Params', num_group=5, criteria_type='kd', format='jpg'): # FLOPs
    # scatter
    # measures = np.log(np.array(measures)+np.e)

    label_x = [((i+1) * 1.0/num_group) for i in range(num_group)]
    max_ = max(measures[factor_type])
    min_ = min(measures[factor_type])
    interval = (max_ - min_) / num_group
    scores_group = {k:[[] for i in range(num_group)] for k in measures.keys()}
    accs_group = {k:[[] for i in range(num_group)] for k in measures.keys()}

    for k, v in measures.items():
        for i, s in enumerate(v):
            idx = int((measures[factor_type][i] - min_) / interval)
            if idx >= num_group: idx = num_group - 1
            for j in range(idx, num_group): # small -> big
                scores_group[k][j].append(s)
                accs_group[k][j].append(accs[i])
                # plt.plot(accs, measures)
    
    accs_group_best = {}
    # scores_group_idx = {}
    for k, vg in scores_group.items():
        accs_group_best[k] = []
        for i,v in enumerate(vg):
            idx = np.argmax(v)
            accs_group_best[k].append(accs_group[k][i][idx])

    for k, v in scores_group.items():
        for i in range(len(v)):
            # scores_group[k][i] = round(return_value(criteria[criteria_type](v[i], accs_group[k][i])), 3)
            scores_group[k][i] = round(return_value(criteria[criteria_type](scores_group[k][i], accs_group[k][i])), 3)


    plt.figure(figsize=(12,10))
    # del scores_group['depth']
    df = pd.DataFrame(scores_group.values(), index=scores_group.keys(), columns=['{:.0%}'.format(i) for i in label_x])
    sns.heatmap(df,annot=True, cmap='YlGnBu', fmt='.3f') # bone_r
    # plt.title(factor_type)

    plt.savefig(os.path.join(path, factor_type+f'.{format}'), bbox_inches='tight')


def get_best_measure(arches, accs, measures, path, measure_type='FLOPs', criteria_type='kd', num_group=5, title='', constraint_type='', rate=0.8, format='jpg'):
    label_x = ['{:.0%}'.format((i+1) * 1.0/num_group) for i in range(num_group)]
    max_ = max(measures[measure_type])
    min_ = min(measures[measure_type])
    interval = (max_ - min_) / num_group
    scores_group_idx = {k:['-1' for _ in range(num_group)] for k in measures.keys()}

    constraint = max(measures[constraint_type]) * rate if constraint_type else 1e10

    for k, v in measures.items():
        for i, s in enumerate(v):
            idx = int((measures[measure_type][i] - min_) / interval)
            if idx >= num_group: idx = num_group - 1
            for j in range(idx, num_group): # small -> big
                if scores_group_idx[k][j] == '-1':
                    scores_group_idx[k][j] = i
                else:
                    best_idx = scores_group_idx[k][j]
                    if (measures[k][i] > measures[k][best_idx]):
                        if constraint_type == '':
                            scores_group_idx[k][j] = i
                        elif measures[constraint_type][i] <= constraint:
                            scores_group_idx[k][j] = i

    best_accs_group = {}
    best_arches_group = {}
    best_scores_group = {}
    for k, idxs in scores_group_idx.items():
        best_accs_group[k] = []
        best_arches_group[k] = []
        best_scores_group[k] = []
        for idx in idxs:
            best_accs_group[k].append(accs[idx])
            best_scores_group[k].append(measures[k][idx])


    plt.figure(figsize=(10,10))
    df = pd.DataFrame(best_accs_group.values(), index=best_accs_group.keys(), columns=label_x) # ['{:.0%}'.format(i) for i in 
    sns.heatmap(df, annot=True, cmap='RdPu', fmt='.1%', annot_kws={'size':17})
    plt.tick_params(labelsize=16)
    savedir = os.path.join(path, f'best_{measure_type}{title}_heatmap.{format}') if constraint_type == '' else os.path.join(path, f'best_{measure_type}{title}_constraint_{constraint_type}_heatmap.jpg')
    plt.savefig(savedir, bbox_inches='tight')
    plt.close()

    if 'combine' in title:
        labels = [0.4, 0.8, 1]
        x = np.arange(len(labels))
        width = 0.06

        fig, ax = plt.subplots(figsize=(10,12))
        rects = []
        scores = ['snip', 'grasp', 'zen']
        bar_scores = {}
        for i, (k, v) in enumerate(best_accs_group.items()):
            key = k.split('_')[0]
            if key not in scores: continue
            bar_scores[k] = [round(i * 100, 1) for i in v]
        
        l = len(bar_scores)
        for i, (k, v) in enumerate(bar_scores.items()):
            y = [v[s] for s in range(len(v)) if (s+1)/5 in labels]
            interval = (i - l//2) * width + i//4 * width
            p1 = ax.barh(x + interval, y, width, label=k)
            rects.append(p1)

        ax.set_yticks(x)
        ax.set_yticklabels(['{:.0%}'.format(i) for i in labels])
        plt.xlim([71, 78.5])

        plt.legend()

        def autolabel(rects):
            for rect in rects:
                width = rect.get_width()
                ax.annotate('{}'.format(width),
                            xy=(width, rect.get_y() + rect.get_height() / 2),
                            xytext=(14, -6),
                            textcoords="offset points",
                            ha='center', va='bottom')

        for i in rects:
            autolabel(i)

        plt.tick_params(labelsize=16)
        fig.savefig(os.path.join(path, f'best_{measure_type}{title}_bar.{format}'), bbox_inches='tight')

    return


def draw_correlation_criteria(arches, accs, measures, path, criteria_type='kd', format='jpg'):
    # varibles = {**measures, 'Accs':accs}
    varibles = {**measures}
    map = np.zeros([len(varibles), len(varibles)])
    names = list(varibles.keys())
    print(names)
    s = list(varibles.values())
    for i in range(len(s)):
        for j in range(len(s)):
            c = criteria[criteria_type](s[j], s[i]) if criteria_type != 'mBR' else criteria[criteria_type](s[j], s[i], measures['FLOPs'])
            map[i, j] = return_value(c)

    # plt.figure(figsize=(40,40))
    plt.figure(figsize=(16,12))
    df = pd.DataFrame(map, index=names, columns=names)
    sns.heatmap(df,annot=True, cmap='YlGnBu', annot_kws={'size':17}) # 
    plt.tick_params(labelsize=16)
    # plt.show()
    plt.savefig(os.path.join(path, criteria_type+'_heatmap')+f'.{format}', bbox_inches='tight')

    return


def draw_measure_histogram(arches, accs, measures, path, title, format='jpg'):
    # row, column = 3, 4
    if len(measures) <= 13:
        row, column = 3, 4 
        plt.figure(figsize=(28,18))
    else:
        row, column = 8, 10
        plt.figure(figsize=(60,50))
    plt.suptitle(title, fontsize=16)
    # figure, ax = plt.subplots(row, column)
    for i, (k, v) in enumerate(measures.items()):
        if k == 'Oracle': continue
        plt.subplot(row, column, i+1)
        plt.ylabel('Density', fontsize=12)
        plt.xlabel('Estimated Score', fontsize=12)
        # t = 'first layer' if i == 0 else f'stage {i}'
        plt.title(k, fontsize=12) # title

        plt.hist(v, bins=50)

    plt.savefig(os.path.join(path, title)+f'.{format}', bbox_inches='tight')
    plt.grid()
    plt.close()

    return


def compute_criteria(arches, accs, scores, title, path, attributes=None):
    info = {}
    for key, f in criteria.items():
        if key == 'BWR@K':
            minn_at_ks = f(scores, accs)
            for i in minn_at_ks:
                # BR@K, WR@K
                (k, minn, br_at_k, maxn, wr_at_k) = i
                info['BR@top{:.1%}'.format(k)] = br_at_k
        elif key == 'P@tbK':
            patks = f(scores, accs)
            for i in patks:
                (ratio, k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk) = i
                info['P@top{:.1%}'.format(ratio)] = p_at_topk
        elif key == 'BR':
            patks = f(scores, accs)
            for i in patks:
                (n, rank_n_ratio) = i
                info['BR={:d}'.format(n)] = rank_n_ratio
        elif key == 'mBR':
            patks = f(scores, accs, attributes)
            info['mBR'] = patks[-1]
        else:
            info[key] = f(scores, accs)
    return info


def print_criteria(info, K=2, path=None, mode='w'):
    hl = ['criteria']
    hl.extend(info.keys())
    
    res = {i:[] for i in info[hl[-1]].keys()}
    for score, value in info.items():
        for key, v in value.items():
            if key == 'depth': continue
            temp = v[K][2] if type(v) == list else v
            res[key].append(round(temp, 3))

    t = PrettyTable()
    t.add_column(hl[0], hl[1:])
    for k, v in res.items():
        t.add_column(k, v)

    print(t)

    with open(path, mode) as f:
        f.write(str(t)+'\n')

    return


def process_measures(measures):
    # if path != None and not os.path.exists(path): os.makedirs(path)
    measures_dict = {k:[] for k in measures[0].keys()}
    # measures_dict = {i:[] for i in ['zen', 'FLOPs', 'Params', 'Oracle']}
    # measures_dict = {k:[] for k in measures[0].keys() if k in ['FLOPs', 'Params', 'depth', 'Oracle']}
    for i in measures:
        for key, value in i.items():
            # if key not in ['FLOPs', 'Params', 'depth', 'Oracle']: continue
            if key == 'synflow':
                value = math.log(value+math.e)
            # if key in ['synflow', 'fisher', 'grad_norm', 'snip', 'grasp', '']:
            #     value = math.log(value+math.e*100000)
            # if key == 'jacob_cov':
            #     value = math.exp(value)
            # if key == 'ntk': value = -1 * value
            # if key not in ['zen', 'FLOPs', 'Params', 'Oracle']: continue
            # if value == math.nan: value = -1e10
            if np.isnan(np.array(value)):
                # value = -1e10
                value = -math.inf
            measures_dict[key].append(value)

    del measures_dict['ntk']
    return measures_dict


def standardize(x):
    x = np.array(x)
    return (x - np.mean(x))/(np.std(x))


def combine_computation(measures, combins=['FLOPs', 'Params'], measures_req=['zen']): # , 'depth'
    from itertools import combinations
    subsets = sum([list(map(list, combinations(combins, i))) for i in range(len(combins) + 1)], [])
    combins = {i:[] for i in combins}
    measures_combins = {'Oracle': measures['Oracle']}
    # measures_req = ['zen'] # 

    for k in combins.keys():
        combins[k] = standardize(measures[k])
        measures_combins[k] = measures[k]

    for m in measures_req:
        measures_combins[m] = measures[m]
        for s in subsets:
            if s == []: continue
            key_comb = '_'.join([m]+s)
            value_comb = standardize(measures[m])
            for j in s: value_comb += combins[j]
            value_comb /= (len(s)+1)
            measures_combins[key_comb] = list(value_comb)

    return measures_combins


def conbine_scores(measures):
    from itertools import combinations
    scores = {i: [] for i in measures if i not in ['FLOPs', 'Params', 'depth', 'Oracle']}
    # scores = {i: [] for i in measures if i not in ['Oracle']}
    subsets = sum([list(map(list, combinations(scores.keys(), i))) for i in range(len(scores) + 1)], [])
    del subsets[0]
    # combins = {i:[] for i in combins}
    scores_combins = {}
    scores_combins_groups = {i: {} for i in range(1, len(scores) + 1)}
    for i in subsets:
        l = len(i)
        name = '+'.join(i)
        # name = str(set(i))
        scores_combins_groups[l][name] = 0

    for k in scores.keys():
        scores[k] = standardize(measures[k])
        
    for s in subsets:
        if s == []: continue
        key_comb = '+'.join(s)
        
        value_comb = sum([scores[i] for i in s]) / len(s)
        
        scores_combins[key_comb] = list(value_comb)
    
    for ks in scores_combins_groups.values():
        for k in ks.keys():
            ks[k] = scores_combins[k]
            # print()
    

    return scores_combins, scores_combins_groups


def draw_multiscores(measures, path, format='jpg'):
    _, scores_combins_groups = conbine_scores(measures)

    best_names = []

    # pre_comb = ['',
    # 'snip+synflow',
    # 'grad_norm+snip+synflow',
    # 'grad_norm+snip+jacob_cov+synflow',
    # 'grad_norm+snip+jacob_cov+synflow+zen',
    # # 'fisher+jacob_cov+plain+synflow+ntk+zen',
    # 'grad_norm+snip+jacob_cov+plain+synflow+ntk+zen',
    # 'fisher+grad_norm+snip+grasp+jacob_cov+plain+synflow+ntk',
    # 'fisher+grad_norm+snip+grasp+jacob_cov+plain+synflow+ntk+zen']

    # pre_comb = ['', # synflow
    # 'snip+synflow',
    # 'grad_norm+snip+synflow',
    # 'grad_norm+snip+jacob_cov+synflow',
    # 'grad_norm+snip+jacob_cov+synflow+zen',
    # 'grad_norm+snip+jacob_cov+plain+synflow+zen',
    # 'fisher+grad_norm+snip+grasp+jacob_cov+plain+synflow',
    # 'fisher+grad_norm+snip+grasp+jacob_cov+plain+synflow+zen']


    pre_comb = ['', # R1 ->
    'snip+synflow',
    'snip+jacob_cov+zen',
    'snip+jacob_cov+synflow+zen',
    'fisher+snip+jacob_cov+plain+synflow',
    'grad_norm+snip+jacob_cov+plain+synflow+zen',
    'fisher+grad_norm+grasp+jacob_cov+plain+synflow+zen',
    'fisher+grad_norm+snip+grasp+jacob_cov+plain+synflow+zen',]
    # pre_comb = []
    num = 1
    # for num in scores_combins_groups.keys():
    while num <= len(scores_combins_groups.keys()):
        if pre_comb == [] or pre_comb[num-1] == '':
            best_name = ''
            # best_acc = 0
            best_mBR = 1
            for k, v in scores_combins_groups[num].items():
                scores_combins_groups[num][k] = criteria['mBR'](v, measures['Oracle'], measures['FLOPs'])[-1]
                if scores_combins_groups[num][k] < best_mBR:
                    best_mBR = scores_combins_groups[num][k]
                    best_name = k
            scores_combins_groups[num] = best_mBR
            best_names.append(best_name)
        else:
            best_name = pre_comb[num-1]
            # best_acc = get_best(scores_combins_groups[num][best_name], measures['Oracle'])[-1] * 100
            # scores_combins_groups[num] = best_acc
            best_mBR = criteria['mBR'](scores_combins_groups[num][best_name], measures['Oracle'], measures['FLOPs'])[-1]
            scores_combins_groups[num] = best_mBR
            best_names.append(best_name)
        num += 1

    plt.figure(figsize=(8, 6))
    # markers = ['o', '^', 's', '*', '+', 'x', 'd', 'P', '1', '2', '3', 'v']
    plt.plot(scores_combins_groups.keys(), scores_combins_groups.values(), linewidth=2.4, marker='*', markerfacecolor='r', markersize=8, label='Combination')
    for x, y in scores_combins_groups.items():
        plt.text(x, y, '%.2f' % y, ha='center', va= 'bottom',fontsize=11)

    # best_acc = max(measures['Oracle'])*100
    # plt.plot(scores_combins_groups.keys(), np.ones(len(scores_combins_groups))*best_acc, linestyle=':', label='Top1 Acc')
    # plt.text(x-0.5, best_acc-0.1, 'y=%.2f' % best_acc, ha='center', va= 'top',fontsize=11)
    # plt.plot(best_names, scores_combins_groups.values(), linewidth=2.4, marker='*', markerfacecolor='r', markersize=8)

    # for i, (k, y) in enumerate(scores_combins_groups.items()):

    # plt.ylabel('Test Acc(%)', fontsize=12)
    plt.ylabel('mBR', fontsize=20)
    plt.xlabel('The number of combinations', fontsize=20)
    plt.tick_params(labelsize=20)

    # plt.axis([-1,10,0,6])

    plt.legend(loc='lower right')
    # plt.title(measure_type)
    plt.ylim([0, 0.2])
    # plt.ylim([75, 79.3])

    plt.savefig(os.path.join(path, f'best_scores_conbination_plot.{format}'), bbox_inches='tight')

    with open(os.path.join(path, 'combinations_scores.txt'), 'w') as f:
        for k, v in zip(best_names, scores_combins_groups.values()):
            f.write(f'{k}: {v}\n')
        # f.write(' '.join(best_names) + '\n')
        # f.write(' '.join([str(i) for i in scores_combins_groups.values()]) + '\n')
        # f.write('Oracle: {}'.format(best_acc) + '\n')

    plt.close()
    return


def analyze_measures_accs(arches, accs, measures_dict, path, fun, analyze_alone=True, *args, **kwargs):
    if path != None and not os.path.exists(path): os.makedirs(path)
    
    if analyze_alone:
        res = {}
        for key, value in measures_dict.items():
            res[key] = fun(arches, accs, value, key, path, *args, **kwargs)
    else:
        res = fun(arches, accs, measures_dict, path, *args, **kwargs)
    
    return res


def load_arch_measure_acc(path):
    data = []
    with open(path,'rb') as f:
        while(1):
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    arches, measures, trainaccs, valaccs, testaccs, FLOPs, Params = [], [], [], [], [], [], []

    for i in data:
        arches.append(i['arch'])
        trainaccs.append(i['trainacc'])
        valaccs.append(i['valacc'])
        testaccs.append(i['testacc'])
        FLOPs.append(i['FLOPs'])
        scale = 1e6 if i['Params'] > 20 else 1
        Params.append(i['Params']/scale)
        
        i['logmeasures']['FLOPs'] = FLOPs[-1]
        i['logmeasures']['Params'] = Params[-1]
        measures.append(i['logmeasures'])
        # i['logmeasures']['depth'] = count_layers_in_model(i['arch'])
        i['logmeasures']['Oracle'] = testaccs[-1]
    
    measures_dict = process_measures(measures)

    return arches, trainaccs, valaccs, testaccs, measures_dict, FLOPs, Params


def get_model_acc_from_bench(opt):
    from nas_201_api import NASBench201API as API
    print('Loading bench201...')
    api = API('/Users/chl/Documents/code/ali/benchmark_score/bench_analyze/NAS-Bench-201-v1_0-e61699.pth', verbose=False) # 201
    accs = []
    flops = []
    params = []
    for idx in api.evaluated_indexes:
        info = api.get_more_info(idx, 'cifar100', None, hp='200', is_random=True)
        info2 = api.query_meta_info_by_index(idx)
        
        accs.append(info['test-accuracy']/100)
        flops.append(info2.all_results[('cifar10-valid', 777)].flop)
        params.append(info2.all_results[('cifar10-valid', 777)].params)

    return accs, flops, params


def main(opt):
    if not os.path.isdir(opt.savedir): os.makedirs(opt.savedir)

    arches, trainaccs, valaccs, testaccs, measures_dict, FLOPs, Params = load_arch_measure_acc(opt.score_path)

    info = analyze_measures_accs(arches, testaccs, measures_dict, None, compute_criteria, attributes=measures_dict['FLOPs'])
    print_criteria(info, path=os.path.join(opt.savedir, 'criteria.txt'))

    measures_req = ['snip', 'grasp', 'zen']
    measures_combine_dict = combine_computation(measures_dict, measures_req=measures_req)
    info = analyze_measures_accs(arches, testaccs, measures_combine_dict, None, compute_criteria, attributes=measures_dict['FLOPs'])
    print_criteria(info, path=os.path.join(opt.savedir, 'criteria_comb.txt'), mode='w')

    criteria_type='BR' # BWR@K kd
    format='pdf'

    draw_multiscores(measures_dict, opt.savedir, format)

    scatter_cost_acc(testaccs, measures_dict['FLOPs'], opt.savedir, 'flops_acc_scatter', format=format)
    scatter_cost_acc(testaccs, measures_dict['Params'], opt.savedir, 'params_acc_scatter', format=format)

    # bench201_accs, bench201_flops, bench201_params = get_model_acc_from_bench(opt)
    # scatter_cost_acc(bench201_accs, bench201_flops, opt.savedir, 'flops_acc_scatter_bench201', format=format)
    # scatter_cost_acc(bench201_accs, bench201_params, opt.savedir, 'params_acc_scatter_bench201', format=format)

    analyze_measures_accs(arches, testaccs, measures_dict, opt.savedir, draw_scatter, title='scatters', analyze_alone=False, format=format)
    analyze_measures_accs(arches, testaccs, measures_dict, os.path.join(opt.savedir, 'scatters'), draw_scatter_individually, analyze_alone=True, format=format)

    analyze_measures_accs(arches, testaccs, measures_dict, opt.savedir, draw_measure_histogram, title='histogram', analyze_alone=False, format=format)

    analyze_measures_accs(arches, testaccs, measures_dict, opt.savedir, draw_correlation_criteria, analyze_alone=False, criteria_type='mBR', format=format)

    num_group = 5 # if i == 'depth' else 10
    # for i in ['FLOPs', 'Params', 'depth']:
    for i in ['FLOPs', 'Params']:
    # for i in ['FLOPs']:

        analyze_measures_accs(arches, testaccs, measures_dict, opt.savedir, draw_params_flops_depth, analyze_alone=False, factor_type=i, num_group=num_group, criteria_type=criteria_type, format=format) # 20

        analyze_measures_accs(arches, testaccs, measures_dict, opt.savedir, get_best_measure, analyze_alone=False, measure_type=i, num_group=num_group, criteria_type=criteria_type, format=format) # 20

        # useless
        # analyze_measures_accs(arches, testaccs, measures_dict, opt.savedir, get_best_measure, analyze_alone=False, measure_type=i, num_group=5, criteria_type=criteria_type, constraint_type='depth', format=format) # 20

        if i != 'depth':
            analyze_measures_accs(arches, testaccs, measures_combine_dict, opt.savedir, get_best_measure, analyze_alone=False, measure_type=i, num_group=num_group, title='_combine', criteria_type=criteria_type, format=format) # 20 constraint_type


    print(f'Save in {opt.savedir}')
    return


if __name__ == '__main__':
    opt = parse_cmd_args(sys.argv)
    main(opt)