#!/usr/bin/env python3
"""Sixteen-seed Moving-F16 confirmation with certified source-kernel charts."""
from __future__ import annotations
import argparse,dataclasses,hashlib,importlib.util,json,math,sys,zipfile
from pathlib import Path
import numpy as np

def load_engine():
    path=Path(__file__).with_name("cner_cnn_mnist_fisher_confirm_v16.py");s=importlib.util.spec_from_file_location("f16_engine",path);m=importlib.util.module_from_spec(s);sys.modules["f16_engine"]=m;s.loader.exec_module(m);return m
E=load_engine()

@dataclasses.dataclass(frozen=True)
class Protocol(E.Protocol):
    protocol_name:str="CNER_CNN_MNIST_MOVING_F16_CONFIRM_V3_0B"
    seeds:int=16
    base_seed:int=65726
    moving_subdivisions_initial:int=1
    moving_subdivisions_max:int=4
    moving_integration_tolerance:float=0.05
    moving_kl_spearman_gate:float=0.80
    moving_kl_median_relative_error_gate:float=0.40
    moving_condition_gate:float=1e6
    moving_rank_min:int=6
    success_required:int=12
    minimum_comparable_seeds:int=14
    wrong_moving_natural_win_ceiling:int=4
    node_cert_locations:int=3
    node_cert_perturbations_per_location:int=8
    node_cert_radius_min:float=0.0025
    node_cert_radius_max:float=0.010
    chart_projection_iterations:int=3
    chart_internal_kernel_target:float=1e-7

def load_protocol(path):
    p=Protocol()
    if path:
        raw=json.loads(Path(path).read_text());valid={f.name for f in dataclasses.fields(p)}
        if set(raw)-valid:raise ValueError("unknown keys: "+repr(sorted(set(raw)-valid)))
        p=Protocol(**raw)
    lock=(p.protocol_name=="CNER_CNN_MNIST_MOVING_F16_CONFIRM_V3_0B" and p.seeds==16 and p.base_seed==65726 and p.primary_metric=="output_fisher_quotient" and p.moving_subdivisions_initial==1 and p.moving_subdivisions_max==4 and p.success_required==12 and p.minimum_comparable_seeds==14 and p.wrong_moving_natural_win_ceiling==4 and p.node_cert_locations==3 and p.node_cert_perturbations_per_location==8 and p.chart_projection_iterations==3)
    if not lock:raise ValueError("v3.0b frozen protocol violated")
    return p

def metric_gate(cert,p):
    return cert["kl_spearman"]>=p.kl_spearman_gate and cert["kl_median_relative_error"]<=p.kl_median_relative_error_gate and cert["gauge_logit_relative_residual"]<=p.gauge_logit_residual_gate and cert["gauge_fisher_relative_residual"]<=p.gauge_fisher_relative_gate and cert["representation_gauge_relative_change"]>=p.representation_gauge_change_gate

def certified_kernel_chart(theta0,J,target_x,target_y,model,spec,p,seed,torch,F,functional_call):
    U,S,Vh=torch.linalg.svd(J,full_matrices=False);threshold=S.max().clamp_min(1e-12)*p.rcond;rank=int((S>threshold).sum());null_dim=theta0.numel()-rank
    if null_dim<p.chart_dim:raise RuntimeError(f"Numerical response kernel dimension {null_dim} < chart_dim {p.chart_dim}")
    Vr=Vh[:rank].T if rank else torch.empty((theta0.numel(),0),device=theta0.device,dtype=theta0.dtype)
    def proj(v):return v-Vr@(Vr.T@v) if rank else v
    v=theta0.detach().clone().requires_grad_(True);loss=F.cross_entropy(functional_call(model,E.params_from_vector(v,spec),(target_x,)),target_y);grad=torch.autograd.grad(loss,v)[0].detach();cols=[];first=proj(-grad)
    if first.norm()<=1e-8:raise RuntimeError("Projected task gradient vanished")
    cols.append(first/first.norm());gen=torch.Generator(device=theta0.device).manual_seed(seed+900001);attempts=0
    while len(cols)<p.chart_dim and attempts<p.chart_dim*40:
        attempts+=1;q=proj(torch.randn(theta0.numel(),generator=gen,device=theta0.device))
        for b in cols:q=q-b*torch.dot(b,q)
        q=proj(q)
        if q.norm()>1e-8:cols.append(q/q.norm())
    if len(cols)!=p.chart_dim:raise RuntimeError("Could not construct certified kernel chart")
    B=torch.stack(cols,dim=1)
    for _ in range(p.chart_projection_iterations):B=proj(B);B,_=torch.linalg.qr(B,mode="reduced")
    residual=float(torch.linalg.norm(J@B)/torch.linalg.norm(J).clamp_min(1e-12));orth=float(torch.linalg.norm(B.T@B-torch.eye(B.shape[1],device=B.device,dtype=B.dtype)))
    if residual>p.chart_internal_kernel_target:raise RuntimeError(f"Internal kernel target failed: {residual}")
    return B.detach(),{"response_rank":rank,"numerical_null_dimension":null_dim,"singular_value_threshold":float(threshold),"kernel_residual":residual,"orthogonality_residual":orth,"projection_iterations":p.chart_projection_iterations}

def moving_action(path,wrong,theta0,B,model,spec,met_x,dev_x,dev_y,p,torch,F,functional_call,subdivisions):
    zzero=torch.zeros(B.shape[1],device=path.device);G0raw=E.output_fisher_raw(theta0,B,model,spec,met_x,torch,functional_call);G0,_=E.regularize_metric(G0raw,p.metric_eigen_floor_relative,torch);reverse=torch.arange(B.shape[1]-1,-1,-1,device=path.device);P=torch.eye(B.shape[1],device=path.device)[:,reverse];G0use=P.T@G0@P if wrong else G0;g0=E.intelligence_gradient(zzero,theta0,B,model,spec,dev_x,dev_y,torch,F,functional_call);scale=float(torch.sqrt((g0@torch.linalg.solve(G0use,g0)).clamp_min(1e-24)))
    action=0.;length=0.;hs=[];conds=[];ranks=[];true_kl=[];quad_kl=[]
    for a,b in zip(path[:-1],path[1:]):
        for j in range(subdivisions):
            aa=a+(b-a)*(j/subdivisions);bb=a+(b-a)*((j+1)/subdivisions);mid=.5*(aa+bb);dz=bb-aa;theta_mid=theta0+B@mid
            Graw=E.output_fisher_raw(theta_mid,B,model,spec,met_x,torch,functional_call);G,meta=E.regularize_metric(Graw,p.metric_eigen_floor_relative,torch);Guse=P.T@G@P if wrong else G
            grad=E.intelligence_gradient(mid,theta0,B,model,spec,dev_x,dev_y,torch,F,functional_call);h=p.h0+float(torch.sqrt((grad@torch.linalg.solve(Guse,grad)).clamp_min(1e-24)))/max(scale,1e-12);dl=float(torch.sqrt((dz@Guse@dz).clamp_min(0)));action+=dl/h;length+=dl;hs.append(h);conds.append(meta["condition"]);eig=torch.linalg.eigvalsh(Graw);ranks.append(int((eig>eig.max().clamp_min(1e-12)*p.metric_eigen_floor_relative).sum()))
            if not wrong:
                ta=theta0+B@aa;tb=theta0+B@bb;true_kl.append(E.mean_output_kl(ta,tb,model,spec,met_x,torch,functional_call));quad_kl.append(float(.5*dz@Graw@dz))
    rho=E.spearman(true_kl,quad_kl) if len(true_kl)>=3 else None;rel=[abs(x-y)/max(abs(x),1e-12) for x,y in zip(true_kl,quad_kl)]
    return {"action":action,"length":length,"effective_capacity":length/max(action,1e-12),"capacity_relative_span":(max(hs)-min(hs))/max(float(np.median(hs)),1e-12),"max_condition":max(conds),"min_effective_rank":min(ranks),"local_kl_spearman":rho,"local_kl_median_relative_error":float(np.median(rel)) if rel else None}

def adaptive_moving(path,wrong,*args):
    p=args[-4];vals={};n=p.moving_subdivisions_initial
    vals[n]=moving_action(path,wrong,*args,subdivisions=n)
    while n<p.moving_subdivisions_max:
        n2=2*n;vals[n2]=moving_action(path,wrong,*args,subdivisions=n2);rel=abs(vals[n2]["action"]-vals[n]["action"])/max(abs(vals[n2]["action"]),1e-12)
        if rel<=p.moving_integration_tolerance:return vals[n2],rel,f"T{n}/T{n2}",True,{str(k):v["action"] for k,v in vals.items()}
        n=n2
    return vals[n],rel,f"T{n//2}/T{n}",False,{str(k):v["action"] for k,v in vals.items()}

def interpolate_path(path,t):
    if len(path)==1:return path[0]
    u=t*(len(path)-1);i=min(int(math.floor(u)),len(path)-2);a=u-i
    return (1-a)*path[i]+a*path[i+1]

def independent_node_certificate(path,seed,control_index,theta0,B,model,spec,met_x,p,torch,functional_call):
    gen=torch.Generator(device=path.device).manual_seed(seed+880001+1009*control_index);true=[];quad=[];conds=[];ranks=[];nodes=[]
    for ni,t in enumerate(np.linspace(0.,1.,p.node_cert_locations)):
        z=interpolate_path(path,float(t));theta=theta0+B@z;Graw=E.output_fisher_raw(theta,B,model,spec,met_x,torch,functional_call);G,meta=E.regularize_metric(Graw,p.metric_eigen_floor_relative,torch);eig=torch.linalg.eigvalsh(Graw);rank=int((eig>eig.max().clamp_min(1e-12)*p.metric_eigen_floor_relative).sum());conds.append(meta["condition"]);ranks.append(rank);nodes.append({"fraction":float(t),"condition":meta["condition"],"effective_rank":rank})
        for j in range(p.node_cert_perturbations_per_location):
            q=torch.randn(B.shape[1],generator=gen,device=path.device);q=q/q.norm().clamp_min(1e-12);k=ni*p.node_cert_perturbations_per_location+j;radius=p.node_cert_radius_min+(p.node_cert_radius_max-p.node_cert_radius_min)*(k/max(p.node_cert_locations*p.node_cert_perturbations_per_location-1,1));dz=radius*q;true.append(E.mean_output_kl(theta,theta+B@dz,model,spec,met_x,torch,functional_call));quad.append(float(.5*dz@Graw@dz))
    rho=E.spearman(true,quad);rel=[abs(a-b)/max(abs(a),1e-12) for a,b in zip(true,quad)]
    return {"samples":len(true),"kl_spearman":rho,"kl_median_relative_error":float(np.median(rel)),"kl_max_relative_error":max(rel),"max_condition":max(conds),"min_effective_rank":min(ranks),"nodes":nodes}

def one_seed(seed,p,outdir,device,deps):
    torch,nn,F,functional_call,DataLoader,Dataset,Subset,datasets,transforms=deps;E.seed_everything(seed,torch);root=str(outdir.parent/"mnist_data");train=datasets.MNIST(root,train=True,download=True,transform=transforms.ToTensor());test=datasets.MNIST(root,train=False,download=True,transform=transforms.ToTensor())
    src,adapt,dev,cap,metric=E.disjoint_train_subsets(train,[p.source_train_size,p.adapt_train_size,p.intelligence_probe_count,p.capability_probe_count,p.metric_probe_count],seed,Subset,torch);src_test=E.take_subset(test,p.source_test_size,seed+1,Subset,torch);src_loader=DataLoader(src,batch_size=p.source_batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0);test_loader=DataLoader(src_test,batch_size=512,shuffle=False,num_workers=0);adapt_loader=DataLoader(adapt,batch_size=p.adapt_batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed+4),num_workers=0)
    model=E.build_components(torch,nn,F)().to(device);E.train_source(model,src_loader,device,p,torch,F);source_acc=E.accuracy_model(model,test_loader,device,torch,shifted=False);spec,nparam=E.vector_spec(model);theta0=E.flatten_model(model,torch).to(device);anchor_x,_=next(iter(DataLoader(src,batch_size=p.anchor_count,shuffle=False)));dev_x,dev_y=next(iter(DataLoader(dev,batch_size=p.intelligence_probe_count,shuffle=False)));cap_x,cap_y=next(iter(DataLoader(cap,batch_size=p.capability_probe_count,shuffle=False)));met_x,_=next(iter(DataLoader(metric,batch_size=p.metric_probe_count,shuffle=False)));tx,ty=next(iter(adapt_loader));anchor_x=anchor_x.to(device);dev_x,dev_y=E.development_batch(dev_x.to(device),torch),dev_y.to(device);cap_x,cap_y=E.shifted_batch(cap_x.to(device),torch),cap_y.to(device);met_x=met_x.to(device);tx,ty=E.shifted_batch(tx.to(device),torch),ty.to(device);batches=[(E.shifted_batch(x.to(device),torch),y.to(device)) for x,y in adapt_loader]
    J=E.response_jacobian(theta0,model,spec,anchor_x,torch,functional_call);B,chart_cert=certified_kernel_chart(theta0,J,tx,ty,model,spec,p,seed,torch,F,functional_call);kernel=chart_cert["kernel_residual"];Graw=E.output_fisher_raw(theta0,B,model,spec,met_x,torch,functional_call);Gf,meta=E.regularize_metric(Graw,p.metric_eigen_floor_relative,torch);reverse=torch.arange(B.shape[1]-1,-1,-1,device=device);P=torch.eye(B.shape[1],device=device)[:,reverse];Gwrong=P.T@Gf@P;cert=E.metric_certification(theta0,B,model,spec,met_x,Graw,p,seed,torch,functional_call);z0=torch.zeros(B.shape[1],device=device);initial=E.loss_at_z(z0,theta0,B,model,spec,cap_x,cap_y,torch,F,functional_call);target=initial*(1-p.capability_loss_reduction_fraction);names=("adam","normalized_sgd","normalized_momentum","sign_gradient","natural_gradient","wrong_fisher_natural_gradient");alg={}
    with torch.no_grad():r0=functional_call(model,E.params_from_vector(theta0,spec),(anchor_x,)).reshape(-1)
    for control_index,name in enumerate(names):
        if name=="natural_gradient":raw=E.executable_path_natural(theta0,B,Gf,model,spec,batches,p,torch,F,functional_call)
        elif name=="wrong_fisher_natural_gradient":raw=E.executable_path_natural(theta0,B,Gwrong,model,spec,batches,p,torch,F,functional_call)
        else:raw=E.executable_path(name,theta0,B,model,spec,batches,p,torch,F,functional_call)
        path,hit,final,step=E.truncate_at_capability(raw,target,theta0,B,model,spec,cap_x,cap_y,torch,F,functional_call);leak=E.path_response_leakage(path,theta0,B,model,spec,anchor_x,r0,torch,functional_call);frozen=E.adaptive_metric_path_action(path,Gf,theta0,B,model,spec,dev_x,dev_y,p,torch,F,functional_call);moving,mrel,mpair,mconv,mvals=adaptive_moving(path,False,theta0,B,model,spec,met_x,dev_x,dev_y,p,torch,F,functional_call);wrong,wrel,wpair,wconv,wvals=adaptive_moving(path,True,theta0,B,model,spec,met_x,dev_x,dev_y,p,torch,F,functional_call);nodecert=independent_node_certificate(path,seed,control_index,theta0,B,model,spec,met_x,p,torch,functional_call);nodegate=nodecert["max_condition"]<=p.moving_condition_gate and nodecert["min_effective_rank"]>=p.moving_rank_min and nodecert["kl_spearman"]>=p.moving_kl_spearman_gate and nodecert["kl_median_relative_error"]<=p.moving_kl_median_relative_error_gate
        alg[name]={"hit_capability":hit,"hit_step":step,"leakage_max":leak["max"],"frozen_action":frozen[0],"frozen_converged":frozen[5],"moving":moving,"moving_integration_relative_change":mrel,"moving_integration_pair":mpair,"moving_converged":mconv,"moving_action_by_subdivisions":mvals,"wrong_moving":wrong,"wrong_moving_converged":wconv,"independent_node_certification":nodecert,"node_metric_gate":nodegate,"admissible":bool(hit and leak["max"]<=p.leakage_relative_gate and frozen[5] and mconv and wconv and nodegate)}
    return {"seed":seed,"source_accuracy":source_acc,"source_gate":source_acc>=p.source_accuracy_gate,"kernel_residual":kernel,"kernel_gate":kernel<=p.kernel_residual_gate,"chart_certification":chart_cert,"source_metric_gate":metric_gate(cert,p),"source_metric_certification":cert,"algorithms":alg}

def summarize(rows,p):
    comp=[];per=[];counts={"moving_natural_wins":0,"frozen_natural_wins":0,"wrong_moving_natural_wins":0,"moving_true_beats_wrong_natural":0}
    for r in rows:
        if not (r["source_gate"] and r["kernel_gate"] and r["source_metric_gate"] and all(a["admissible"] for a in r["algorithms"].values())):continue
        comp.append(r);a=r["algorithms"];mw=min(a,key=lambda n:a[n]["moving"]["action"]);fw=min(a,key=lambda n:a[n]["frozen_action"]);ww=min(a,key=lambda n:a[n]["wrong_moving"]["action"]);counts["moving_natural_wins"]+=mw=="natural_gradient";counts["frozen_natural_wins"]+=fw=="natural_gradient";counts["wrong_moving_natural_wins"]+=ww=="natural_gradient";counts["moving_true_beats_wrong_natural"]+=a["natural_gradient"]["moving"]["action"]<a["wrong_fisher_natural_gradient"]["moving"]["action"];nc=a["natural_gradient"]["independent_node_certification"];per.append({"seed":r["seed"],"moving_winner":mw,"frozen_winner":fw,"wrong_moving_winner":ww,"natural_moving_action":a["natural_gradient"]["moving"]["action"],"wrong_natural_moving_action":a["wrong_fisher_natural_gradient"]["moving"]["action"],"natural_frozen_action":a["natural_gradient"]["frozen_action"],"chart_kernel_residual":r["chart_certification"]["kernel_residual"],"chart_null_dimension":r["chart_certification"]["numerical_null_dimension"],"natural_node_max_condition":nc["max_condition"],"natural_node_min_rank":nc["min_effective_rank"],"natural_node_kl_spearman":nc["kl_spearman"],"natural_node_kl_median_relative_error":nc["kl_median_relative_error"]})
    n=len(comp);gate=n>=p.minimum_comparable_seeds and counts["moving_natural_wins"]>=p.success_required and counts["moving_true_beats_wrong_natural"]>=p.success_required and counts["wrong_moving_natural_wins"]<=p.wrong_moving_natural_win_ceiling
    status="MOVING_F16_V30B_CONFIRMED_RESTRICTED_ORDERING" if gate else ("MOVING_F16_V30B_INADMISSIBLE" if n<p.minimum_comparable_seeds else "MOVING_F16_V30B_ORDERING_NOT_SUPPORTED")
    return {"scientific_status":status,"seeds":len(rows),"fully_comparable":n,"required_comparable":p.minimum_comparable_seeds,"counts":counts,"moving_f16_confirmation_gate":gate,"per_seed":per,"claim_boundary":"Sixteen-new-seed confirmation of six executable paths under a certified source-response-kernel chart, pointwise moving output-Fisher pullback, and jointly moving capacity dual norm. Metric certification uses 24 fixed perturbations at three path locations. It remains chart-fixed rather than a moving full-space projector or cross-chart quotient geometry, and does not establish arbitrary-path global minimality or a universal learning law."}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--protocol",type=Path);ap.add_argument("--output",type=Path,default=Path("cner_cnn_mnist_moving_f16_confirm_v3_0b_results"));ap.add_argument("--no-download",action="store_true");args,_=ap.parse_known_args();p=load_protocol(args.protocol);deps=E.import_torch();torch=deps[0];device=torch.device("cuda" if torch.cuda.is_available() else "cpu");print(f"[preflight] device={device} torch={torch.__version__} seeds={p.seeds}",flush=True);args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for i in range(p.seeds):
        seed=p.base_seed+i;print(f"[Moving-F16 v3.0b seed {i+1}/{p.seeds}] {seed}",flush=True);r=one_seed(seed,p,args.output,device,deps);rows.append(r);(args.output/f"seed_{seed}.json").write_text(json.dumps(r,indent=2)+"\n")
    sm=summarize(rows,p);result={"protocol":dataclasses.asdict(p),"provenance":{"script_sha256":sha(__file__)},"summary":sm,"seeds":rows};(args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");(args.output/"protocol.json").write_text(json.dumps(dataclasses.asdict(p),indent=2)+"\n");(args.output/"REPORT.md").write_text("# Moving-F16 confirmation v3.0b\n\n```json\n"+json.dumps(sm,indent=2)+"\n```\n");zp=args.output.parent/(args.output.name+".zip")
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for f in args.output.rglob("*"):
            if f.is_file():z.write(f,f.relative_to(args.output.parent))
    print("="*96);print(json.dumps(sm,indent=2));print("Results ZIP:",zp.resolve())
    if not args.no_download:
        try:
            from google.colab import files;files.download(str(zp))
        except Exception:pass
if __name__=="__main__":main()
