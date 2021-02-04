## Implement
* [global-objective loss function](https://github.com/TsungHuaLee/TMB/blob/master/utils/global_objective.py)
* multiple instance learning + noise learning
    * [modified DivideMix](https://github.com/TsungHuaLee/TMB/blob/master/DivideMix.ipynb)
    * [ensenble prediction + gmm noisy filter + mixmatch](https://github.com/TsungHuaLee/TMB/blob/master/MixMatch.ipynb)
    * [ensenble prediction + gmm noisy filter + mean teacher](https://github.com/TsungHuaLee/TMB/blob/master/MT-low-rank.ipynb)

## Experiment Result
* Tumor Mutation Burden

|  | TMB-H/TMB-L |
| -------- | -------- |
| AUROC     | 0.88    |


* Single Gene Mutation

|  | APC | KRAS | TP53 |
| -------- | -------- | -------- | -------- |
| AUROC     | 0.69    | 0.55     | 0.65    |


## Reference
1.	E.Eban, M.Schain, A.Mackey, A.Gordon, R. A.Saurous, andG.Elidan, “Scalable learning of non-decomposable objectives,” Proc. 20th Int. Conf. Artif. Intell. Stat. AISTATS 2017, vol. 54, 2017.
2.	L.Berrada, A.Zisserman, andM. P.Kumar, “Smooth loss functions for deep top-k classification,” arXiv, pp. 1–25, 2018.
3. 	S. E. L. F.Nsembling, “Self : Learning To Filter Noisy Labels,” pp. 1–16, 2020.
4. 	J.Li, R.Socher, andS. C. H.Hoi, “Dividemix: Learning with noisy labels as semi-supervised learning,” arXiv, pp. 1–14, 2020.
5.	J. N.Kather et al., “Pan-cancer image-based detection of clinically actionable genetic alterations,” bioRxiv, pp. 1–19, 2019, doi: 10.1101/833756.
6.	H.Li, Y. F.Wang, R.Wan, S.Wang, T. Q.Li, andA. C.Kot, “Domain generalization for medical imaging classification with linear-dependency regularization,” arXiv, no. NeurIPS, pp. 1–12, 2020.
7.	P.Izmailov, D.Podoprikhin, T.Garipov, D.Vetrov, andA. G.Wilson, “Averaging weights leads to wider optima and better generalization,” 34th Conf. Uncertain. Artif. Intell. 2018, UAI 2018, vol. 2, pp. 876–885, 2018.
