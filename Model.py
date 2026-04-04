import func
import os
import math
import parameter as para
import rsome as rso
from rsome import ro
from rsome import cpt_solver as cpt
import pandas as pd
from gurobipy import GRB
from gurobipy import quicksum
import gurobipy as gp
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
random.seed(1)

class Data:
    def __init__(self, instance_name, vessel, berth, alpha, service_time):
        
        self.berth = berth
        self.alpha = alpha 
        self.service_time = service_time
        
        self.ins = pd.read_csv(f'{para.data_path}{instance_name}')
        # self.ins = mergeDelta(self.ins, delta_result, delta_classify_name)
        # self.ins.to_csv(f'{para.data_path}{instance_name}', index=False)
        
        self.vessel = len(self.ins) if len(self.ins) <= vessel else vessel
        self.ETA = {i+1: v for i, v in enumerate(self.ins['ETA'][:self.vessel])}
        self.ATA = {i+1: v for i, v in enumerate(self.ins['ATA'][:self.vessel])}
        self.PTA = {i+1: v for i, v in enumerate(self.ins['PTA'][:self.vessel])}
        self.T = self.ins['T'].iloc[0]
        self.weight = {i: 1 for i in range(1, self.vessel + 1)}
        self.s = {i: self.service_time for i in range(1, self.vessel + 1)}
        self.I = [i for i in range(1, self.vessel + 1)] 
        self.K = [f'b({i})' for i in range(1, berth + 1)]
        
        self.raw_delta_a = {i+1: v for i, v in enumerate(self.ins['delta'][:self.vessel])}
        self.delta_a = {i: abs(v) for i,v in self.raw_delta_a.items()}
        self.delta_a_positive = {i: v if v > 0 else 0 for i,v in self.raw_delta_a.items()}
        self.delta_a_square = {i: v**2 if v > 0 else 0 for i,v in self.raw_delta_a.items()}
        self.delta_s = {i : 0 for i in self.I}
        
        self.ETA[0] = 0
        self.ATA[0] = 0
        self.PTA[0] = 0
        self.s[0] = 0
        self.delta_a[0] = 0
        self.delta_a_positive[0] = 0
        self.delta_a_square[0] = 0
        self.delta_s[0] = 0
        
        for k in self.K:
            self.ETA[k] = self.T
            self.ATA[k] = self.T
            self.PTA[k] = self.T
            self.s[k] = 0

        self.N = [0] + self.I + self.K
        self.Incoming = [0] + self.I
        self.Outgoing = self.I + self.K
        self.A = [(0,j) for j in self.I] + [(i,j) for i in self.I for j in self.I if i != j] + [(i,j) for i in self.I for j in self.K]
        
        self.sol_d_eta = Solution(p="DM(ETA)")
        self.sol_d_pta = Solution(p="DM(PTA)")
        self.sol_rs = Solution(p="RS")
        self.sol_rsp = Solution(p="RSP")
        self.sol_rss = Solution(p="RSS")
        self.sol_rso = Solution(p="RSO")
        
        self.indicator_map = {
            "TotWait": "tot_waiting",
            "MaxWait": "max_waiting",
            "QuanWait": "quan_waiting",
            "StdWait": "std_waiting",
            
            "VarWait25": "var_waiting25",
            "CvarWait25": "cvar_waiting25",
            "VarWait50": "var_waiting50",
            "CvarWait50": "cvar_waiting50",
            "VarWait75": "var_waiting75",
            "CvarWait75": "cvar_waiting75",
            "VarWait95": "var_waiting95",
            "CvarWait95": "cvar_waiting95",
            
            "NumWait": "waiting_num",
            "NumTWait": "exceed_waiting_num",
            "CPU": "cpu"
        }
        self.metrics = list(self.indicator_map.keys())
        
        
class Solution:
    def __init__(self, p):
        self.problem = p
        self.x_value = {}
        self.tau_value = {}
        self.sum_tau = 0
        self.m_obj = 0
        self.cpu = 0
        self.paths = []
        self.berthing_time = {}
        self.berthing_time_pta = {}
        self.waiting_list = {}
        self.waiting_list_pta = {}
        self.tot_waiting = 0
        self.max_waiting = 0
        self.quan_waiting = 0
        self.std_waiting = 0
        self.var_waiting25 = 0
        self.cvar_waiting25 = 0
        self.var_waiting50 = 0
        self.cvar_waiting50 = 0
        self.var_waiting75 = 0
        self.cvar_waiting75 = 0
        self.var_waiting95 = 0
        self.cvar_waiting95 = 0
        self.exceed_waiting_num = 0
        self.waiting_num = 0
        self.nberth = 0
        self.m_obj_pta_rs = 0
        self.That = 0
        
def evaluateSolution(sol, m, x, tau):
    sol.x_value = {(i,j): x[i,j].x for i,j in data.A if x[i,j].x > 0.1}
    sol.tau_value = {i: func.twoDecimal(tau[i].x) for i in data.I}
    sol.sum_tau = func.twoDecimal(sum(tau[i].x for i in data.I))
    
    sol.m_obj = func.twoDecimal(m.ObjVal)
    sol.cpu = func.twoDecimal(m.Runtime)
    sol.paths = buildPath(sol.x_value)
    calWaitingTime(sol)
    sol.nberth = len(sol.paths)
    sol.That = func.twoDecimal(sol.m_obj * (1 + data.alpha))

def writeResult(models):
    base_info = {
        "instance": instance_name,
        "#vessel": data.vessel,
        "#berth": data.berth,
        "alpha": data.alpha,
        "That": models["DM(PTA)"].That,
        "service": data.service_time
    }
    for model_name, sol in models.items():
        row = {
            "Model": model_name,
            **base_info,
            "Obj": sol.m_obj,
            "SumTau": sol.m_obj_pta_rs,
            # "Tau": sol.sum_tau
            }
        for k in data.metrics:
            row[k] = getattr(sol, data.indicator_map[k])
        results.append(row)
    return results

def getSummaryPerformance(df, excel_filename):
    summary = df.groupby("Model")[data.metrics].mean()
    summary["Count"] = df.groupby("Model")[data.metrics[0]].count()
    summary = summary.reset_index()
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="RawData", index=False)

def getGroupSummaryPerformance(df, excel_filename, selected_models):
    # figure_path = "../Result/Alpha1018/Figure/"
    # df = pd.read_csv("../Result/Alpha1018/CSV/2023-01-09.csv")
    summary = df[df["Model"].isin(selected_models)].groupby(["alpha", "Model"])[data.metrics].mean().reset_index()
    def normalize(group):
        base = group.loc[group["Model"] == "DM(ETA)", data.metrics]
        if base.empty:
            return group  
        base_values = base.iloc[0]
        group[data.metrics] = round(group[data.metrics] / base_values, 3)
        return group
    summary = summary.groupby("alpha", group_keys=False).apply(normalize)
    # summary["Mean"] = round(summary[data.metrics[:-1]].mean(axis=1), 3)
    
    def highlight_min_by_alpha(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in data.metrics:
            min_idx = df[col].idxmin()
            styles.loc[min_idx, col] = "font-weight: bold; color: red"
        return styles

    # styled = summary.groupby("alpha", group_keys=False).apply(highlight_min_by_alpha)
    # styled_df = summary.style.apply(lambda _: styled, axis=None)
    # styled_df.to_excel(f"{csv_path}normalized_style.xlsx", index=False)
    
    # with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
    #     styled_df.to_excel(writer, sheet_name="Summary", index=False)
    #     df.to_excel(writer, sheet_name="RawData", index=False)
    
    metrics = [c for c in summary.columns if c not in ["alpha", "Model"]]
    summary_sorted = summary.sort_values(["alpha", "Model"])
    
    # ==== user parameters ====
    n_cols = 3          # number of sugfigures each row
    single_w, single_h = 4, 4   # weight and height of each figure
    dpi = 600           
    # =========================

    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(single_w * n_cols, single_h * n_rows),
        dpi=dpi
    )
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model, group in summary_sorted.groupby("Model"):
            ax.plot(group["alpha"], group[metric], marker="o", markersize=3, label=model)
        ax.set_title(metric)
        ax.set_xlabel("alpha")
        ax.set_ylabel("value")
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == 0:
            ax.legend()
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f"{figure_path}metrics_grid.jpg", dpi=dpi)
    plt.show()
    
    for metric in metrics:
        plt.figure(figsize=(7, 5), dpi=400)
        for model, group in summary_sorted.groupby("Model"):
            plt.plot(group["alpha"], group[metric],
                     marker="o", markersize=4, label=model)
        plt.xlabel("alpha")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{figure_path}{metric}.jpg")
        plt.show()
        plt.close()
    
def outgoing(i):
    return [arc[1] for arc in data.A if arc[0] == i]

def incoming(i):
    return [arc[0] for arc in data.A if arc[1] == i]

def buildPath(x_value):
    succ = {}
    for i, j in x_value.keys():
        succ.setdefault(i, []).append(j)
    paths = []
    stack = [[0]]  
    while stack:
        path = stack.pop()
        last = path[-1]
        if last in succ: 
            for nxt in succ[last]:
                stack.append(path + [nxt])
        else:  
            paths.append(path)
    return paths

def calWaitingTime(sol):
    t = {} #waiting time at each vessel under ATA
    t_pta = {} #waiting time at each vessel under PTA
    for path in sol.paths:
        tt = 0 #berthing time at each vessel
        tt_pta = 0
        for i in range(len(path) - 2):
           pre_node = path[i]
           node = path[i + 1]
           
           t[node] = max(data.ATA[node], tt + data.s[pre_node])
           t_pta[node] = max(data.PTA[node], tt_pta + data.s[pre_node])
           
           tt = t[node]
           tt_pta = t_pta[node]
           
           sol.berthing_time[node] = func.twoDecimal(t[node])
           sol.berthing_time_pta[node] = func.twoDecimal(t_pta[node])
           
           t[node] -= data.ATA[node]
           t_pta[node] -= data.PTA[node]
           sol.waiting_list[node] = func.twoDecimal(t[node]) 
           sol.waiting_list_pta[node] = func.twoDecimal(t_pta[node]) 
    t_list = list(t.values())
    sol.m_obj_pta_rs = sum(t_pta.values())
    sol.tot_waiting = func.twoDecimal(np.mean(t_list))
    sol.max_waiting = func.twoDecimal(max(t.values()))
    sol.quan_waiting = func.twoDecimal(np.percentile(t_list, 75))
    sol.std_waiting = func.twoDecimal(np.std(t_list, ddof=0))
    
    levels = [0.25, 0.50, 0.75, 0.95]
    arr = np.array(t_list)
    for c in levels:
        var_val = np.percentile(arr, (1 - c) * 100)
        cvar_val = arr[arr <= var_val].mean()
        setattr(sol, f"var_waiting{int(c*100)}", func.twoDecimal(var_val))
        setattr(sol, f"cvar_waiting{int(c*100)}", func.twoDecimal(cvar_val))

    sol.waiting_num = func.twoDecimal(sum(1 for i in t_list if i > 0) / data.vessel)
    sol.exceed_waiting_num = func.twoDecimal(sum(1 for i in t_list if (i - 1) > 0) / data.vessel)

def write(content, clear = False):
    mode = "w" if clear else "a"
    with open(txt_filename, mode, encoding="utf-8") as f:
        if isinstance(content, list): #if content is a list
            print(content)
            f.writelines([line if line.endswith("\n") else line + "\n" for line in content])
        else:
            print(content)
            f.write(content + "\n")

def getStorePath(folder_name):
    csv_path = f"../Result/{folder_name}/CSV/"
    txt_path = f"../Result/{folder_name}/TXT/"
    figure_path = f"../Result/{folder_name}/Figure/"
    func.createFolder(csv_path)
    func.createFolder(txt_path)
    func.createFolder(figure_path)
    func.createFolder(para.data_path)
    return csv_path, txt_path, figure_path
    
def printMessage(sol):
    write(f'\n\n============ {instance_name} {sol.problem}: ============')
    write(f"alpha = {data.alpha} and serviceTime = {data.s}")
    write("Detailed Results: ")
    for path in sol.paths:
        write(f"{path}: ")
        for i in range(len(path) - 1):
            node = path[i]
            if i == 0: continue
            write(f"At node {node}: Tau({sol.tau_value[node]}), ETA({data.ETA[node]}), PTA({data.PTA[node]}), ATA({data.ATA[node]}), Delta({func.twoDecimal(data.raw_delta_a[node])}), BT({sol.berthing_time[node]}), WT({sol.waiting_list[node]})")
            # write(f"At node {node}: Tau({sol.tau_value[node]}), ATA({data.ATA[node]}), ETA({data.ETA[node]}), PTA({data.PTA[node]}), Diff({func.twoDecimal(data.PTA[node]-data.ETA[node])}), WTPTA({sol.waiting_list_pta[node]}), WT({sol.waiting_list[node]}), BTPTA({sol.berthing_time_pta[node]}), BT({sol.berthing_time[node]}), s({data.s[node]})")
    write('===============================================================')

def printSummaryMessage(instance_name, models: dict):
    write(f'\n\n====== {instance_name} summary: ======\n')
    for label, attr in data.indicator_map.items():
        values = [getattr(sol, attr) for sol in models.values()]
        write(f"{label}: {' -- '.join(map(str, values))}")
    write(f'\n====== {instance_name} end ======')

def sortFiles(files):
    dates = pd.to_datetime([f.replace(".csv", "") for f in files], errors="coerce")
    return [f for _, f in sorted(zip(dates, files))]

def getResidualMetric(err, quant=0.90):
    #err: ATA - ETA 
    e = np.asarray(err, dtype=float).ravel()
    mask = np.isfinite(e)
    e = e[mask]
    n = e.size
    out = dict(n=int(n))

    if n == 0:
        base = dict(mede=0, medae=0, rmse=0, p90ae=0, early_rate=0, late_rate=0, mean=0, std=0, skew=0, kurt=0, q25=0, q50=0, q75=0)
        out.update(base)
        return out

    ae = np.abs(e)
    out["mede"]   = float(np.median(e))
    out["medae"]  = float(np.median(ae))
    out["rmse"]   = float(np.sqrt(np.mean(e**2)))
    out["p90ae"]  = float(np.quantile(ae, quant))
    out["early_rate"] = float(np.mean(e < 0))
    out["late_rate"] = float(np.mean(e > 0))
    out["mean"] = float(np.mean(e))
    out["std"]  = float(np.std(e, ddof=1))
    out["skew"] = float(st.skew(e, bias=False))
    out["kurt"] = float(st.kurtosis(e, bias=False))  # excess kurtosis
    out["q25"]  = float(np.quantile(e, 0.25))
    out["q50"]  = float(np.quantile(e, 0.50))  # median again
    out["q75"]  = float(np.quantile(e, 0.75))
    return out

def getPCA(df):
    feature_cols = ["mede", "medae", "rmse", "p90ae", "early_rate", "late_rate", "mean", "std", "skew", "kurt", "q25", "q50", "q75"]
    X = df[feature_cols].values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_std).ravel()
    df_out = df.copy()
    df_out["PCA"] = scores
    return df_out, pca

def getDelta(instance_name, classify_name):
    # classify_name = para.due_last_port
    train_data_name = instance_name.replace(".csv", "").rsplit("-", 1)[0]
    train_data = pd.read_csv(f"../Data/Training/{train_data_name}.csv")
    results = []
    mvcs_result = {}
    for name, group in train_data.groupby(classify_name):
        z = group[para.arrival_delay].values.reshape(-1, 1)
        if z.shape[0] == 1: 
            mvcs_result[name] = z[0,0]
        else:
            mvcs_result[name] = mvcs(z)
    # ==== PCA ===
    for name, group in train_data.groupby(classify_name):
        z = group[para.arrival_delay].values.reshape(-1, 1)
        metrics = getResidualMetric(group[para.arrival_delay].values, quant=0.90)
        metrics[classify_name] = name
        results.append(metrics)
    group_result = pd.DataFrame(results)
    group_result = group_result.fillna(0)
    result_pca, pca_model = getPCA(group_result)
    pca_result = dict(zip(result_pca["DUE_LAST_PORT"], result_pca["PCA"]))
    # ====================
    return pca_result, mvcs_result

def mergeDelta(ins, delta_result, delta_classify_name):
    ins["delta"] = ins[delta_classify_name].map(delta_result).fillna(0)
    return ins

def mvcs(z, s=None, r=None, p1=2, p2=2, display=True):
    N, L = z.shape #L is the dimension of z, N is the number of in-sample
    model = ro.Model()
    q = model.dvar(L)
    u = model.dvar(N)
    v = model.dvar(N)
    Q = model.dvar((L, L))
    
    if s is None:
        side = np.zeros(N)
    else:
        S = s.shape[1] #dimension of vector s
        P = model.dvar((L, S))
        side = s@P.T
        
    if L == 1:
        model.max(Q[0, 0]) #Q is a scalar
    else:
        model.max(rso.rootdet(Q))
        
    # model.max(rso.rootdet(Q))
    model.st((1/N) * u.sum() <= 1)
    if isinstance(p2, tuple): 
        model.st((1/L) * rso.power(v, *p2) <= u) #p2 is a tuple, each element of p2 as an input parameter
    else:
        model.st((1/L) * rso.power(v, p2) <= u) #rso.power(v,p2) = v^p2
    l_norm = lambda x: rso.norm(x, p1)  #define a function 'l_norm': p-norm of x
    for n in range(N):
        model.st(l_norm(Q@z[n] - q - side[n]) <= v[n])
    if r is not None:
        model.st(Q << np.diag(r ** (-1)))  # regularization constraint            
    if display:
        msg = f"Sample data:       {N} records x {L} inputs \n"
        msg += "Side information:  "
        msg += "None\n" if s is None else f"{S} features\n"
        msg += f"Norm type:         l{p1}-norm\n"
        msg += f"Deviation penalty: power={p2}\n"
        print(msg)
        print(model.do_math())
    model.solve(cpt, display=display)
    if s is None:
        Q_inv = np.linalg.inv(Q.get())
        return (Q_inv @ q.get())[0]                             # outputs with no side information
    else:
        return q.get(), P.get(), Q.get()                    # outputs with inside information

def defineDomain(m, x, v):
    m.addConstrs(x.sum(i,'*') == 1 for i in data.I)
    m.addConstrs(x.sum('*', i) == 1 for i in data.I)
    m.addConstrs(quicksum(x[j,i] for j in incoming(i)) <= 1 for i in data.K)
    m.addConstrs(v[i,j,0] == 0 for i,j in data.A)
    m.addConstrs(v[i,j,l] <= x[i,j] for i,j in data.A for l in data.Outgoing)
    m.addConstrs(quicksum(v[0,j,l] for j in outgoing(0)) - quicksum(v[i,l,l] for i in incoming(l)) == 0 for l in data.Outgoing)
    m.addConstrs(quicksum(v[0,j,l] for j in outgoing(0)) - quicksum(x[i,l] for i in incoming(l)) == 0 for l in data.Outgoing)
    m.addConstrs(quicksum(v[l,j,l] for j in outgoing(l)) == 0 for l in data.I)
    m.addConstrs(quicksum(v[i,j,l] for j in outgoing(i)) - quicksum(v[j,i,l] for j in incoming(i)) == 0 for l in data.Outgoing for i in data.I if i != l)

def DM(sol, arrival_time):
    m = gp.Model()
    # m.Params.OutputFlag = 0
    tau = m.addVars(data.I, vtype = GRB.CONTINUOUS, name = 'tau')
    x = m.addVars(data.A, vtype = GRB.BINARY, name = 'x')
    v = m.addVars(data.A, data.N, vtype = GRB.CONTINUOUS, name = 'v')
    m.setObjective(quicksum(data.weight[i]*tau[i] for i in data.I))
    m.addConstrs(
        quicksum(v[i,k,l] for i in incoming(k)) * arrival_time[k] + 
        quicksum(data.s[i] * (v[i,j,l] - v[i,j,k]) for i,j in data.A) - arrival_time[l] <= 
        tau[l] for k in data.Incoming for l in data.I
        )
    # m.addConstrs(
    #     quicksum(v[i,k,l] for i in incoming(k)) * arrival_time[k] + 
    #     quicksum(data.s[i]*(v[i,j,l] - v[i,j,k]) for i,j in data.A) <= 2*data.T
    #     for k in data.Incoming for l in data.K
    #     )
    defineDomain(m, x, v)
    m.optimize()
    evaluateSolution(sol, m, x, tau)
    printMessage(sol)
    m.dispose()

def RS(sol, ahat, shat, delta_a, delta_s):
    m = gp.Model()
    # m.Params.OutputFlag = 0
    tau = m.addVars(data.I, vtype = GRB.CONTINUOUS, name = 'tau')
    x = m.addVars(data.A, vtype = GRB.BINARY, name = 'x')
    v = m.addVars(data.A, data.N, vtype = GRB.CONTINUOUS, name = 'v')
    eta = m.addVars(data.I, vtype = GRB.CONTINUOUS, name = 'eta')
    m.setObjective(quicksum(eta[i] for i in data.I))
    m.addConstrs(eta[l] >= delta_a[l] + 
    quicksum(delta_a[i] * (v[i,j,l] - v[i,j,k]) for i,j in data.A) for l in data.I for k in data.Incoming)
    m.addConstrs(
        quicksum(v[i,k,l] for i in incoming(k)) * ahat[k] + 
        quicksum(shat[i] * (v[i,j,l] - v[i,j,k]) for i,j in data.A) - ahat[l] <= 
        tau[l] for k in data.Incoming for l in data.I
        )
    # m.addConstrs(
    #     quicksum(v[i,k,l] for i in incoming(k)) * ahat[k] + 
    #     quicksum(shat[i] * (v[i,j,l] - v[i,j,k]) for i,j in data.A) <= data.T
    #     for k in data.Incoming for l in data.K
    #     )
    m.addConstr(quicksum(data.weight[i] * tau[i] for i in data.I) <=  data.sol_d_pta.m_obj * (1 + data.alpha))
    defineDomain(m, x, v)
    m.write("test.lp")
    m.optimize()
    v_value = {(i,j,k): v[i,j,k].x for (i,j) in data.A for k in data.N if v[i,j,k].x > 0}
    eta_value = {(i): eta[i].x for i in data.I}
    evaluateSolution(sol, m, x, tau)
    printMessage(sol)
    m.dispose()

def goModels():
    
    print(f'====== {instance_name} DM(ETA): ======')
    DM(data.sol_d_eta, data.ETA)
    print(f'====== {instance_name} DM(PTA): ======')
    DM(data.sol_d_pta, data.PTA)
    print(f'====== {instance_name} RS: ======')
    RS(data.sol_rs, data.PTA, data.s, data.delta_a, data.delta_s)
    # print(f'====== {instance_name} RSP: ======')
    # RS(data.sol_rsp, data.PTA, data.s, data.delta_a_positive, data.delta_s)
    # print(f'====== {instance_name} RSS: ======')
    # RS(data.sol_rss, data.PTA, data.s, data.delta_a_square, data.delta_s)
    models = {"DM(ETA)": data.sol_d_eta, "DM(PTA)": data.sol_d_pta, "RS": data.sol_rs}
    # models = {"DM(ETA)": data.sol_d_eta, "DM(PTA)": data.sol_d_pta, "RS": data.sol_rs, "RSP": data.sol_rsp, "RSS": data.sol_rss}
    return models

def writeCS(models):
    printSummaryMessage(instance_name, models)
    results = writeResult(models)
    pd.DataFrame(results).to_csv(f'{csv_path}{instance_name}', index=False)
    return results
   
if __name__ == "__main__":
    results = []
    csv_path, txt_path, figure_path = getStorePath(folder_name="Alpha1018") #define the stored results path ../Result/folder_name/CSV
    instance_list = sortFiles(os.listdir(para.instance_path))
    solved_instance_list = os.listdir(txt_path)
    delta_classify_name = para.due_last_port #prediction deviation based on it

    instance_idx = 0
    
    # for instance_name in instance_list:
    #     instance_idx += 1
    #     instance_name = '2022-12-01.csv'
    #     # if (instance_name.replace(".csv", ".txt") in solved_instance_list): continue
    #     # pca_delta_result, mvcs_delta_result = getDelta(instance_name, delta_classify_name)
    #     print(f"=== {instance_idx}-th {instance_name} ===")
    #     delta_result = {i + 1:1 for i in range(19)}
    #     data = Data(instance_name, vessel=8, berth=3, delta_result = delta_result, alpha=0.04, service_time=12)
    #     DM(data.sol_d_eta, data.ETA)
        # break
    # ============================================
    for instance_name in instance_list:
        instance_idx += 1
        instance_name = '2022-12-01.csv'
        # if (instance_name.replace(".csv", ".txt") in solved_instance_list): continue
        # pca_delta_result, mvcs_delta_result = getDelta(instance_name, delta_classify_name)
        txt_filename = txt_path + instance_name.replace(".csv", ".txt")
        write("", clear = True)
        for alpha_value in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            for service_value in [12]:
                print(f"=== {instance_idx}-th {instance_name} ===")
                data = Data(instance_name, vessel=8, berth=3, alpha=alpha_value, service_time=service_value)
                models = goModels()
                results = writeCS(models)
        if instance_name == '2023-01-09.csv': break
        # break
    results = pd.DataFrame(results)
    # getGroupSummaryPerformance(results, f'{csv_path}performance.xlsx', selected_models = ["DM(ETA)", "DM(PTA)", "RS", "RSP", "RSS"])
    # getGroupSummaryPerformance(results, f'{csv_path}performance.xlsx', selected_models = ["DM(ETA)", "DM(PTA)", "RS"])
    # ============================================

