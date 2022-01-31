
from math import inf
import sys, os
import json
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_score_factor import load_arch_measure_acc
from utils.config_utils import load_config
from utils.train_utils import set_seed


def parse_cmd_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--savedir', type=str, default='./bench_analyze/results')
    parser.add_argument('--source_path', type=str, default='./search_space/results/vcnn_cifar100_seed777_dlrandom_dlinfo1_7168.p')
    # /Users/chl/Documents/code/ali/benchmark_score/search_space/results/vcnn_cifar100_seed777_dlrandom_dlinfo1_3584.p
    parser.add_argument('--search_space_config', type=str, default='./configs/search_space/cifar100_SuperResConvKXKX.yaml')
    parser.add_argument('--factor', type=str, default=None)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def preprare_trainning_data_(file_name, val_num=50):
    with open(file_name, 'r') as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    X_all, Y_all  = np.array(X_all), np.array(Y_all)
    # X_all, Y_all  = torch.Tensor(X_all), torch.Tensor(Y_all)

    index_list = np.arange(X_all.shape[0])
    np.random.shuffle(index_list)
    X_all = X_all[index_list, :]
    Y_all = Y_all[index_list]

    X_train, Y_train, X_val, Y_val = X_all[:-val_num], Y_all[:-val_num], X_all[-val_num:], Y_all[-val_num:]
    return X_train, Y_train, X_val, Y_val


def split_cv_fold(n_fold, X, Y):
    X_train_set_list = []
    X_cv_set_list = []
    Y_train_set_list = []
    Y_cv_set_list = []

    n = X.shape[0]
    n_val = round(n // n_fold)
    n_train = n - n_val
    index_list = np.arange(n)

    for i in range(n_fold):
        np.random.shuffle(index_list)
        X_train_set_list.append(X[index_list[0:n_train], :])
        Y_train_set_list.append(Y[index_list[0:n_train]])
        X_cv_set_list.append(X[index_list[n_train:], :])
        Y_cv_set_list.append(Y[index_list[n_train:]])

    return X_train_set_list, Y_train_set_list, X_cv_set_list, Y_cv_set_list


def mre(y_predicted, y_true):
    acc = np.mean(np.abs((y_predicted - y_true)/y_true))
    return acc


def do_one_experiment(X_train, Y_train, X_test, Y_test):
    X_train_set_list, Y_train_set_list, X_cv_set_list, Y_cv_set_list = split_cv_fold(10, X_train, Y_train)

    # reg_lam_list = 10 ** np.linspace(-4, 2, 20, endpoint=True)
    reg_lam_list = [10] # ** np.linspace(-4, 2, 20, endpoint=True)
    val_acc_list = []
    pbar1 = tqdm(total=len(reg_lam_list), postfix='reg')
    for reg_lam in reg_lam_list:

        param = {'reg_lambda': reg_lam,
                 'max_depth': 10, 'eta': 1.0, 'gamma': 0.001, 'subsample': 0.8,
                 'eval_metric': 'rmse',
                 'objective': 'reg:squarederror',
                 }
        num_round = 100

        cv_acc_list = []
        for X_train_set, Y_train_set, X_cv_set, Y_cv_set in zip(X_train_set_list, Y_train_set_list,
                                                                X_cv_set_list, Y_cv_set_list):
            # print(f'reg: {cnt}/{len(X_train_set_list)}')
            xgb_train = xgb.DMatrix(X_train_set, label=Y_train_set.ravel())
            xgb_test = xgb.DMatrix(X_cv_set)

            bst = xgb.train(param, xgb_train, num_round)
            the_pred = bst.predict(xgb_test).reshape(Y_cv_set.shape)
            the_acc = np.sqrt(np.mean((the_pred - Y_cv_set) ** 2))
            cv_acc_list.append(the_acc)

        avg_cv_acc = np.mean(cv_acc_list)
        val_acc_list.append(avg_cv_acc)

        pbar1.update()

    best_index = np.argmin(val_acc_list)
    best_reg_lam = reg_lam_list[best_index]

    best_solver_ensemble_pred_list = []
    pbar2 = tqdm(total=len(X_train_set_list), postfix='acc')
    for X_train_set, Y_train_set, X_cv_set, Y_cv_set in zip(X_train_set_list, Y_train_set_list,
                                                            X_cv_set_list, Y_cv_set_list):
        # print(f'acc: {cnt}/{len(X_train_set_list)}')
        param = {'reg_lambda': best_reg_lam,
                 'max_depth': 10, 'eta': 1.0, 'gamma': 0.001, 'subsample': 0.8,
                 'eval_metric': 'rmse',
                 'objective': 'reg:squarederror',
                 }
        num_round = 100

        xgb_train = xgb.DMatrix(X_train_set, label=Y_train_set.ravel())
        xgb_test = xgb.DMatrix(X_test)

        bst = xgb.train(param, xgb_train, num_round)
        the_pred = bst.predict(xgb_test).reshape((-1, 1))

        best_solver_ensemble_pred_list.append(the_pred)
        pbar2.update()

    best_solver_ensemble_pred = np.concatenate(best_solver_ensemble_pred_list, axis=1)
    best_solver_ensemble_pred = np.mean(best_solver_ensemble_pred, axis=1)

    best_test_acc = np.sqrt(np.mean((best_solver_ensemble_pred - Y_test) ** 2))
    # ypred_contribs = bst.predict(xgb_test, pred_contribs=True)
    # importance = ypred_contribs.mean(0)
    

    # print(f'best_reg_lam={best_reg_lam:4g}, best_test_acc={best_test_acc:4g}')

    return best_reg_lam, best_test_acc, bst


def traintest(X_train, Y_train, X_test, Y_test, reg_lam, feature_names):
    param = {'reg_lambda': reg_lam,
                'max_depth': 10, 'eta': 1.0, 'gamma': 0.001, 'subsample': 0.8,
                'eval_metric': 'rmse',
                'objective': 'reg:squarederror',
                'importance_type': 'weight'
                }
    # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model = xgb.XGBRegressor(**param)
    
    # xgb_train = xgb.DMatrix(X_train, label=Y_train.ravel())
    # xgb_test = xgb.DMatrix(X_test)

    model.fit(X_train, Y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)
    the_acc = mre(ans, Y_test)
    # model.get_booster().feature_names = feature_names

    return the_acc, model


def do_one_exp(X_train, Y_train, X_test, Y_test, feature_names):
    # XGBoost训练过程
    reg_lam_list = 10 ** np.linspace(-4, 2, 20, endpoint=True)
    # reg_lam_list = [10] # ** np.linspace(-4, 2, 20, endpoint=True)

    val_acc_list = []
    model_list = []
    for reg_lam in reg_lam_list:
        the_acc, model = traintest(X_train, Y_train, X_test, Y_test, reg_lam, feature_names)
        val_acc_list.append(the_acc)
        model_list.append(model)

    best_index = np.argmin(val_acc_list)
    best_reg_lam = reg_lam_list[best_index]
    best_test_acc = val_acc_list[best_index]
    best_model = model_list[best_index]

    # 显示重要特征
    # plot_importance(model)
    # plt.show()

    return best_reg_lam, best_test_acc, best_model # , bst


def class_dict(config):
    # CHANNEL = {1}
    CLASS = {'ConvKXBNRELU': 1}
    for i in config.blocktype_list:
        CLASS[i] = len(CLASS) + 1
    # CLASS = {i: idx + 1 for idx, i in enumerate(config.blocktype_list)}
    return CLASS


def encode_input(archs, config):
    CLASS = class_dict(config)
    # [class, in, out, s, k, L, btr]
    max_len = max([len(i) for i in archs])
    max_len_idx = np.argmax([len(i) for i in archs])

    be = 1 if len(config.blocktype_list) == 1 else 0
    end = 6 if len(config.btr_list) == 1 else 7
    res = [[list(stage.values())[be:end] for stage in arch] for arch in archs]
    feature_names = [list(i.keys())[be:end] for i in archs[max_len_idx]]
    if len(config.kernel_list) == 1:
        for i in range(len(feature_names)):
            feature_names[i].remove('k')
            feature_names[i] = [f'S{i}'+j for j in feature_names[i]]
    feature_names = sum(feature_names, [])

    for i in range(len(res)):
        # res[i][0].append(1) # 2D
        for j in range(len(res[i])):
            if len(config.kernel_list) == 1:
                k_p = 3 if len(config.blocktype_list) == 1 else 4
                a = res[i][j].pop(k_p)
            if be == 0: res[i][j][0] = CLASS[res[i][j][0]]
        
        for _ in range(max_len-len(res[i])):
            res[i].append([0 for m in range(len(res[i][-1]))])
        
        res[i] = sum(res[i], [])

    return np.array(res), feature_names


def preprare_trainning_data(file_name, search_space_config, val_rate=0.2):
    # [
    # {'class': 'ConvKXBNRELU', 'in': 3, 'out': 64, 's': 1, 'k': 3},
    # {'class': 'SuperResConvKXKX', 'in': 64, 'out': 64, 's': 1, 'k': 3, 'L': 2, 'btn': 1},
    # {'class': 'SuperResConvKXKX', 'in': 64, 'out': 128, 's': 2, 'k': 3, 'L': 2, 'btn': 1},
    # {'class': 'SuperResConvKXKX', 'in': 128, 'out': 256, 's': 2, 'k': 3, 'L': 2, 'btn': 1},
    # {'class': 'SuperResConvKXKX', 'in': 256, 'out': 512, 's': 2, 'k': 3, 'L': 2, 'btn': 1},
    # ]

    config = load_config(search_space_config)
    arches, _, valaccs, testaccs, measures_dict, FLOPs, Params = load_arch_measure_acc(file_name)
    # measures_dict.pop('depth')
    measures_dict.pop('FLOPs')
    measures_dict.pop('Params')

    X_all, feature_names = encode_input(arches, config)
    Y_all = {k: np.array(v) for k, v in measures_dict.items()}

    index_list = np.arange(X_all.shape[0])
    np.random.shuffle(index_list)

    X_all = X_all[index_list, :]
    Y_all = {k: v[index_list] for k, v in Y_all.items()}  # Y_all[index_list]

    val_num = int(len(X_all) * val_rate)
    X_train, X_val = X_all[:-val_num], X_all[-val_num:]

    Y_train = {k: v[:-val_num] for k, v in Y_all.items()}
    Y_val = {k: v[-val_num:] for k, v in Y_all.items()}
    return X_train, Y_train, X_val, Y_val, feature_names


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=18)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_importance_all(models_dict, feature_names, opt, format='jpg'):
    row, column = 3, 3
    fig = plt.figure(figsize=(18,14))
    # plt.suptitle('Importance', fontsize=16)
    
    for i, (measure, bst) in enumerate(models_dict.items()):
        bst.get_booster().feature_names = feature_names
        if measure == 'Oracle':
            fig = plt.figure(figsize=(18,14))
            plot_importance(bst, title=measure, importance_type='gain', show_values=False)
            plt.savefig(os.path.join(opt.savedir, f'importance_best.{format}'), bbox_inches='tight')
            break
        ax = fig.add_subplot(row, column, i+1)
        # bst.feature_names = feature_names
        plot_importance(bst, title=measure, ax=ax, importance_type='gain', show_values=False) # .set_yticklabels(feature_names) # bst.get_booster()
        if i == len(models_dict)-2:
            plt.savefig(os.path.join(opt.savedir, f'importance.{format}'), bbox_inches='tight')
            plt.close()

    return


def plot_importance_radar(models_dict, feature_names, opt, format='jpg'):
    importances = {}
    for i, (measure, bst) in enumerate(models_dict.items()):
        # bst.get_booster().feature_names = feature_names
        # importance_ = bst.get_booster().get_score(
        #     importance_type='weight', fmap='')
        
        importance = bst.feature_importances_
        # importance[np.where(importance>0.6)] = 0.6
        importances[measure] = importance
    idxs = np.where(sum(importances.values()))
    for k, v in importances.items():
        importances[k] = v[idxs]
    feature_names = np.array(feature_names)[idxs]

    N = len(feature_names)
    plt.xticks(fontsize=14)
    theta = radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(figsize=(8,6), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    
    for i, (measure, importance) in enumerate(importances.items()):
        if measure == 'Oracle':
            continue
        axs.plot(theta, importance, label=measure) # , color=color
        axs.fill(theta, importance, alpha=0.1) # , facecolor=color

    legend = plt.legend(loc=(0.9, .95),
                    labelspacing=0.1, fontsize='large')
    axs.set_varlabels(feature_names)
    plt.savefig(os.path.join(opt.savedir, f'importance_radar.{format}'), bbox_inches='tight')
    plt.close()

    # best
    # N = len(feature_names)
    # theta = radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(figsize=(8,6), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
    axs.plot(theta, importances['Oracle'], label='Oracle') # , color=color
    axs.fill(theta, importances['Oracle'], alpha=0.25) # , facecolor=color
    legend = plt.legend(loc=(0.9, .95),
                labelspacing=0.1, fontsize='large')
    axs.set_varlabels(feature_names)
    plt.savefig(os.path.join(opt.savedir, f'importance_best_radar.{format}'), bbox_inches='tight')

    return


def main(opt):
    set_seed(opt.seed)
    if not os.path.exists(opt.savedir): os.makedirs(opt.savedir)

    best_test_acc_dict = {}
    best_reg_lam_dict = {}
    models_dict = {}
    # for exp_count in range(1):
    #     print(f'Exp: {exp_count}')

    X_train, Y_train_dict, X_test, Y_test_dict, feature_names = preprare_trainning_data(opt.source_path, opt.search_space_config, val_rate=0.2)
    # X_train, Y_train, X_test, Y_test = preprare_trainning_data_(stage1_file, val_num=50)

    for measure in tqdm(Y_train_dict.keys()):
        Y_train, Y_test = Y_train_dict[measure], Y_test_dict[measure]
        # best_reg_lam, best_test_acc, bst = do_one_experiment(X_train, Y_train, X_test, Y_test)
        best_reg_lam, best_test_acc, bst = do_one_exp(X_train, Y_train, X_test, Y_test, feature_names)

        models_dict[measure] = bst
        best_reg_lam_dict[measure] = best_reg_lam
        best_test_acc_dict[measure] = best_test_acc

    plot_importance_all(models_dict, feature_names, opt, format='pdf')
    plot_importance_radar(models_dict, feature_names, opt, format='pdf')

    with open(os.path.join(opt.savedir, 'measures_reg.txt'), 'w') as f:
        for k in best_test_acc_dict.keys():
            info = f'{k}: mean_best_acc={best_test_acc_dict[k]:.2%}, mean_best_reg_lam={best_reg_lam_dict[k]:4g}'
            print(info)
            f.write(info+'\n')
        # print(f'mean_best_acc={mean_best_acc:4g}, mean_best_reg_lam={mean_best_reg_lam:4g}')

    return


if __name__ == '__main__':
    opt = parse_cmd_args(sys.argv)
    main(opt)
