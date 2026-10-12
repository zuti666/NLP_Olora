from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


@lru_cache(maxsize=None)
def _load_learner_from_file(rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    file_path = repo_root / rel_path
    module_name = f"_dynamic_learner_{file_path.stem.replace('-', '_')}"
    spec = spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Learner from {file_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Learner


def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models_CL.simplecil import Learner
    elif name == "ogd":
        from models_Project2.models_Full.OGD import Learner
    elif name == "ogd_collector":
        from models_Project2.models_Full.OGD_collector import Learner
    elif name == "ewc":
        from models_CL.EWC import Learner
    elif name == "ewcon":
        from models_CL.EWCon import Learner
    elif name in {"ewcon_probe", "ewcon_mechanism", "ewcon_fqa"}:
        from models_CL.EWCon_Probe import Learner
    elif name in {
        "fishersam_ewc_lambda",
        "fisher_sam_ewc_lambda",
        "fishersam-ewc-lambda",
        "fisher-sam-ewc-lambda",
        "fsam_ewc_lambda",
    }:
        Learner = _load_learner_from_file("models_CL/FisherSAM-EWC-lambda.py")
    elif name in {
        "fishersam_ewc_lambda_kfac",
        "fisher_sam_ewc_lambda_kfac",
        "fishersam-ewc-lambda-kfac",
        "fisher-sam-ewc-lambda-kfac",
        "fsam_ewc_lambda_kfac",
        "fishersam_ewc_lambda_kfacfishersam",
    }:
        Learner = _load_learner_from_file("models_CL/FisherSAM-EWC-lambda-KFACFisherSAM.py")
    elif name in {"ewc_gam_projected", "ewcon_gam_projected", "ewc_gam_proj"}:
        from models_CL.ewc_gam_projected import Learner
    elif name in {
        "ewc_gam_merged_intask_euc",
        "ewc_gam_merge_intask_euc",
        "ewc_gam_intask_euc",
        "ewcon_gam_merged_intask_euc",
    }:
        from models_CL.ewc_gam_merged_intask_euc import Learner
    elif name in {
        "ewc_gam_mergedintask_multifisher_norm",
        "ewc_gam_merged_intask_multifisher_norm",
        "ewc_gam_multifisher_norm",
        "ewcon_gam_mergedintask_multifisher_norm",
    }:
        from models_CL.ewc_gam_mergedintask_multifisher_norm import Learner
    elif name in {
        "ewc_gam_mergedintask_multifisher_norm_weightmergethenfisher",
        "ewc_gam_merged_intask_multifisher_norm_weightmergethenfisher",
        "ewc_gam_multifisher_norm_weightmergethenfisher",
        "ewc_gam_weightmergethenfisher",
        "ewc_gam_wmergethenfisher",
    }:
        from models_CL.ewc_gam_mergedintask_multifisher_norm_weightmergethenfisher import Learner
    elif name in {"ewcon_probe_gam_factored", "ewcon_gam_factored", "ewcon_factored"}:
        from models_CL.EWConProbe_GAMFactored import Learner
    elif name in {"ewc_fishershape", "fs_ewc", "ewcon_fishershape", "fishershape_ewc"}:
        from models_CL.EWConProbe_FisherShape import Learner
    elif name in {
        "ewcon_probe_fishershape_factored",
        "ewcon_fishershape_factored",
        "ewcon_fs_factored",
        "fs_ewc_factored",
    }:
        from models_CL.EWConProbe_FisherShapeFactored import Learner
    elif name in {"ewcon_rwp_gaussian", "ewcon_rwp_gauss"}:
        from models_CL.EWCon_RWP_Gaussian import Learner
    elif name in {"ewcon_rwp_fisher", "ewcon_fisher"}:
        from models_CL.EWCon_RWP_Fisher import Learner
    elif name in {"fo_so", "fo_so_noise", "ogd_noiseinject", "ogd_noise_inject"}:
        from models_Project2.models_Full.OGD_noiseInject import Learner
    elif name in {"ogd_noiseinject_gaussian", "ogd_noise_inject_gaussian", "fo_so_noise_gaussian"}:
        from models_Project2.models_Full.OGD_noiseInject_gaussian import Learner
    elif name in {"ogd_rwp_gaussian", "ogd_rwp", "fo_so_rwp_gaussian"}:
        from models_Project2.models_Full.OGD_RWP_Gaussian import Learner
    elif name in {"rwp_gaussian", "fo_so_rwp_gaussian_noogd"}:
        from models_CL.RWP_Gaussian import Learner
    elif name in {"ogd_rwp_fisher", "fo_so_rwp_fisher", "ogd_rwp_fisher_noise"}:
        from models_Project2.models_Full.OGD_RWP_Fisher import Learner
    elif name in {"rwp_fisher", "fo_so_rwp_fisher_noogd"}:
        from models_CL.RWP_Fisher import Learner
    elif name in {"ogd_rwp_fisher_2", "fo_so_rwp_fisher_2", "ogd_rwp_fisher_v2"}:
        from models_Project2.models_Full.OGD_RWP_Fisher_2 import Learner
    elif name in {"ogd_fisher3", "ogd_rwp_fisher_3", "fo_so_rwp_fisher_3", "ogd_fisher_3"}:
        from models_Project2.models_Full.OGD_Fisher3 import Learner
    elif name in {"rwp_fisher_2", "fo_so_rwp_fisher_2_noogd", "rwp_fisher2"}:
        from models_CL.RWP_Fisher_2 import Learner
    elif name in {"fo_so_noise_optimizer", "fo_so_noise_opt", "ogd_noiseinject_optimizer", "ogd_noise_inject_optimizer"}:
        from models_Project2.models_Full.OGD_noiseInject_optimizer import Learner
    elif name == "si":
        from models_CL.SI import Learner
    elif name == "si_ewc":
        from models_CL.SI_EWC import Learner
    elif name == "si_ewcon":
        from models_CL.SI_EWCon import Learner
    elif name == "gpm":
        from models_Project2.models_Full.GPM import Learner
    elif name == "gpm_ewc":
        from models_Project2.models_Full.GPM_EWC import Learner
    elif name == "agem":
        from models_CL.AGEM import Learner
    elif name == "gem":
        from models_Project2.models_Full.GEM import Learner
    elif name in {"gem_noise", "gem-noise", "gemnoise"}:
        from models_Project2.models_Full.GEM_noise import Learner
    elif name in {"gem_noise_pert", "gem-noise-pert", "gemnoisepert"}:
        from models_Project2.models_Full.GEM_noise_pert import Learner
    elif name == "gem_ewc":
        from models_Project2.models_Full.GEM_EWC import Learner
    elif name == "piece":
        from models_CL.PIECE import Learner
    elif name == "fopng":
        from models_Project2.models_Full.FOPNG import Learner
    elif name == "flad":
        from models_Project2.models_Full.FLAD import Learner
    elif name in {"flad_filterogdfisher", "flad_filter_ogd_fisher", "flad-ogd-fisher"}:
        from models_Project2.models_Full.FLAD_filterogdfisher import Learner
    elif name in {"flad_approxminte", "flad_fd", "flad-fd"}:
        from models_Project2.models_Full.FLAD_approxminte import Learner
    elif name in {
        "flad_secondpreviousfishercontrol",
        "flad_secondprevious_fisher_control",
        "flad_spfc",
        "flad-secondpreviousfishercontrol",
    }:
        from models_Project2.models_Full.FLAD_SecondPreviousFisherControl import Learner
    elif name in {"sam_ogd", "samogd"}:
        from models_CL.SAM_OGD import Learner
    elif name in {"gam_ogd_fisher", "gam_ogd_fishermethod", "gamogdfisher"}:
        from models_Project2.models_Full.GAM_OGD_FisherMethod import Learner
    elif name == "geolora":
        from models.GeoLoRA import Learner
    # loraBased
    elif name in {"cllora_aligned", "cllora"}:
        from models_LoRAbasedCL.cllora_aligned import Learner
    elif name == "sdlora":
        from models_LoRAbasedCL.sdlora import Learner
    elif name == "ogd_lora":
        from models_LoRAbasedCL.OGD_LoRA2 import Learner
    elif name in {"ogd_fisher_lora", "ogd-fisher-lora", "ogdfisherlora", "ogd_fishe_lora", "ogd-fishe-lora"}:
        from models_LoRAbasedCL.OGD_Fisher_LoRA import Learner
    elif name == "ewc_lora":
        from models_LoRAbasedCL.EWC_LoRA_AB import Learner
    elif name == "ewc_lora_ab":
        from models_LoRAbasedCL.EWC_LoRA_AB import Learner
    elif name == "ewclora":
        from models_LoRAbasedCL.ewclora_origin import Learner
    elif name == "ewc_yaoyue_lora_liying":
        from models_LoRAbasedCL.EWC_Yaoyue_LoRA_liying import Learner
    elif name == "ewclora_youyue_github":
        from ref.ewclora_youyue_github import Learner
    elif name in {"ewclora_youyue_fitarchitechture", "ewclora_youyue_fitarchitecture"}:
        from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner
    elif name in {
        "ewclora_youyue_fitarchitechture_gam",
        "ewclora_youyue_fitarchitecture_gam",
        "ewclora_youyue_fitarch_gam",
    }:
        from models_LoRAbasedCL.ewclora_youyue_fitArchitechture_gam import Learner
    elif name in {
        "as1_normfisher_gam",
        "as1_normfisher",
        "ewclora_normfisher_gam",
    }:
        from models_LoRAbasedCL.as1_normfisher_gam import Learner
    elif name in {
        "as1_normfisher_fishershape",
        "as1_normfisher_fs",
        "ewclora_normfisher_fishershape",
        "ewclora_fishershape",
        "fs_ewclora",
    }:
        from models_LoRAbasedCL.as1_normfisher_fishershape import Learner
    elif name == "as1_normfisher_fishershape_complementarity":
        from models_LoRAbasedCL.as1_normfisher_fishershape_complementarity import Learner
    elif name == "as1_normfisher_fishershape_kfac":
        from models_LoRAbasedCL.as1_normfisher_fishershape_kfac import Learner
    elif name == "as1_normfisher_fishershape_richardson":
        from models_LoRAbasedCL.as1_normfisher_fishershape_richardson import Learner
    elif name == "as1_normfisher_fishershape_central":
        from models_LoRAbasedCL.as1_normfisher_fishershape_central import Learner
    elif name == "as1_normfisher_fishershape_freshlora_merge":
        from models_LoRAbasedCL.as1_normfisher_fishershape_freshlora_merge import Learner
    elif name in {
        "as1_normfisher_fishershape_freshlora_merge_factor",
        "as1_normfisher_fishershape_freshlora_merge-factor",
        "freshlora_merge_factor",
        "merge_factor",
        "merge-factor",
    }:
        from models_LoRAbasedCL.as1_normfisher_fishershape_freshlora_merge_factor import Learner
    elif name in {
        "as1_normfisher_fishershape_decoupled",
        "as1_normfisher_fs_decoupled",
        "ewclora_normfisher_fishershape_decoupled",
        "fs_decoupled_ewclora",
    }:
        from models_LoRAbasedCL.as1_normfisher_fishershape_decoupled import Learner
    elif name in {
        "as1_normfisher_fishershape_efficient",
        "as1_normfisher_efficientfs",
        "as1_normfisher_efs",
        "ewclora_efficient_fishershape",
        "efficient_fishershape_ewclora",
        "lookfs_ewclora",
    }:
        from models_LoRAbasedCL.as1_normfisher_fishershape_efficient import Learner
    elif name in {
        "as1_normfisher_fishershape_multidirectional",
        "as1_normfisher_multidirectional_fs",
        "as1_normfisher_pmdfs",
        "ewclora_multidirectional_fishershape",
        "periodic_multidirectional_fs_ewclora",
        "pmdfs_ewclora",
    }:
        from models_LoRAbasedCL.as1_normfisher_fishershape_multidirectional import Learner
    elif name in {
        "as1_normfisher_sam",
        "as1_normfisher_sam_ref",
        "ewclora_normfisher_sam",
        "sam_ewclora",
        "sam_ewc_lora",
    }:
        from models_LoRAbasedCL.as1_normfisher_sam import Learner
    elif name in {
        "as1_normfisher_gam_intask_projected",
        "as1_normfisher_gam_intask_project",
        "as1_normfisher_gam_intask_lambda_project",
        "ewclora_normfisher_gam_intask_project",
        "ewclora_normfisher_gam_intask_lambda_project",
    }:
        from models_LoRAbasedCL.as1_normfisher_gam_intask_lambda_project import Learner
    elif name in {
        "as1_normfisher_gam_intask",
        "ewclora_normfisher_gam_intask",
    }:
        from models_LoRAbasedCL.as1_normfisher_gam_intask import Learner
    elif name in {
        "as1_normfisher_gam_intask_damp",
        "ewclora_normfisher_gam_intask_damp",
    }:
        from models_LoRAbasedCL.as1_normfisher_gam_intask_damp_weightmergethenfisher import Learner
    elif name in {
        "as1_normfisher_gam_intask_damp_project",
        "as1_normfisher_gam_intask_damp_wmerge_project",
        "as1_normfisher_gam_intask_damp_weightmergethenfisher_project",
        "ewclora_normfisher_gam_intask_damp_project",
        "ewclora_normfisher_gam_intask_damp_wmerge_project",
        "ewclora_normfisher_gam_intask_damp_weightmergethenfisher_project",
    }:
        Learner = _load_learner_from_file(
            "models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher-project.py"
        )
    elif name in {
        "fishersam_ewclora_lambda",
        "fisher_sam_ewclora_lambda",
        "fishersam_ewc_lora_lambda",
        "fisher_sam_ewc_lora_lambda",
        "fsam_ewclora_lambda",
        "fsam_ewc_lora_lambda",
    }:
        from models_LoRAbasedCL.FisherSAM_EWCLoRA_lambda import Learner
    elif name in {
        "fishersam_ewclora_lambda_project",
        "fisher_sam_ewclora_lambda_project",
        "fishersam_ewc_lora_lambda_project",
        "fisher_sam_ewc_lora_lambda_project",
        "fsam_ewclora_lambda_project",
        "fsam_ewc_lora_lambda_project",
    }:
        from models_LoRAbasedCL.FisherSAM_EWCLoRA_lambda_project import Learner
    elif name in {
        "gam_ewclora_lambda",
        "gam_ewc_lora_lambda",
    }:
        from models_LoRAbasedCL.GAM_EWCLoRA_lambda import Learner
    elif name in {
        "gam_ewclora_lambda_norm",
        "gam_ewc_lora_lambda_norm",
        "ewclora_normfisher_gam_lambda",
    }:
        from models_LoRAbasedCL.GAM_EWCLoRA_lambda_norm import Learner
    elif name in {
        "gam_ewclora_lambda_project",
        "gam_ewc_lora_lambda_project",
    }:
        from models_LoRAbasedCL.GAM_EWCLoRA_lambda_project import Learner
    elif name in {
        "gam_ewclora_ncl",
        "gam_ewc_lora_ncl",
        "ewclora_ncl",
    }:
        Learner = _load_learner_from_file("models_LoRAbasedCL/GAM_EWCLoRA_NCL.py")
    elif name in {
        "gam_ewclora_ncl_kfac",
        "gam_ewc_lora_ncl_kfac",
        "ewclora_ncl_kfac",
    }:
        Learner = _load_learner_from_file("models_LoRAbasedCL/GAM_EWCLoRA_NCL-KFAC.py")
    elif name in {
        "ncl_kfac_lora",
        "ncl_kfa_lora",
        "ncl_kfac",
        "ncl_kfa",
    }:
        Learner = _load_learner_from_file("models_LoRAbasedCL/NCL-KFA.py")
    elif name == "fo_so_lora":
        from models_LoRAbasedCL.OGD_EWC_LoRA import Learner
    elif name == "si_lora":
        from models_LoRAbasedCL.SI_AB_LoRA import Learner
    elif name == "seqlora":
        from models_LoRAbasedCL.seqlora import Learner
    elif name in {"seqlora_ogd", "seqlora-ogd", "seq_lora_ogd", "seqloraogd"}:
        Learner = _load_learner_from_file("models_LoRA_drift-difussion/SeqLoRA_OGD.py")
    elif name in {
        "seqlora_ogd_new",
        "seqlora-ogd-new",
        "seq_lora_ogd_new",
        "seqloraogdnew",
        "seqlora_ogd_deltaw",
        "seqlora-ogd-deltaw",
    }:
        Learner = _load_learner_from_file("models_LoRA_drift-difussion/SeqLoRA_OGD_new.py")
    elif name in {"seqlora_gem", "seqlora-gem", "seq_lora_gem", "seqloragem", "seqloragrem"}:
        Learner = _load_learner_from_file("models_LoRA_drift-difussion/SeqLoRA_GEM.py")
    elif name in {
        "seqlora_gem_new",
        "seqlora-gem-new",
        "seq_lora_gem_new",
        "seqloragemnew",
        "seqlora_gem_deltaw",
        "seqlora-gem-deltaw",
    }:
        Learner = _load_learner_from_file("models_LoRA_drift-difussion/SeqLoRA_GEM_new.py")
    elif name in {"seqlora_ewc", "seqlora-ewc", "seq_lora_ewc", "seqloraewc", "seqlora_ewc_yaoyue", "seqlpra_ewc"}:
        Learner = _load_learner_from_file("models_LoRA_drift-difussion/SeqLoRA_EWC_yaoyue.py")
    elif name in {
        "seqlora_ewc_new",
        "seqlora-ewc-new",
        "seq_lora_ewc_new",
        "seqloraewcnew",
        "seqlora_ewc_deltaw",
        "seqlora-ewc-deltaw",
    }:
        Learner = _load_learner_from_file("models_LoRA_drift-difussion/SeqLoRA_EWC_new.py")
    elif name == "inclora":
        from models_LoRAbasedCL.inclora import Learner
    elif name == "olora":
        from models_LoRAbasedCL.olora import Learner
    elif name == "inflora":
        from models_LoRAbasedCL.inflora import Learner
    elif name == "eflora":
        from models_LoRAbasedCL.eflora import Learner
    elif name == "infogam_lora":
        from models_LoRAbasedCL.infogam_lora import Learner
    elif name == "infogamlora":
        from models_LoRAbasedCL.infogam_lora import Learner
    elif name == "infobudget_gam_lora":
        from models_LoRAbasedCL.infobudget_gam_lora import Learner
    elif name == "infobudgetgamlora":
        from models_LoRAbasedCL.infobudget_gam_lora import Learner
    elif name == "infobudget_lora":
        from models_LoRAbasedCL.infobudget_gam_lora import Learner
    elif name == "infobudget_gam_lora_subnce":
        from models_LoRAbasedCL.infobudget_gam_lora_subnce import Learner
    elif name == "infobudget_gam_lora_nce":
        from models_LoRAbasedCL.infobudget_gam_lora_subnce import Learner
    elif name == "infobudget_subnce_lora":
        from models_LoRAbasedCL.infobudget_gam_lora_subnce import Learner
    elif name == "infobudget_gam_rankfreeze_lora":
        from models_LoRAbasedCL.infobudget_gam_rankfreeze_lora import Learner
    elif name == "infobudget_rankfreeze_lora":
        from models_LoRAbasedCL.infobudget_gam_rankfreeze_lora import Learner
    elif name == "infobudgetgamlorarankfreeze":
        from models_LoRAbasedCL.infobudget_gam_rankfreeze_lora import Learner
    elif name == "infobudget_gam_rankrecycle_lora":
        from models_LoRAbasedCL.infobudget_gam_rankrecycle_lora import Learner
    elif name == "infobudget_rankrecycle_lora":
        from models_LoRAbasedCL.infobudget_gam_rankrecycle_lora import Learner
    elif name == "infobudgetgamlorarankrecycle":
        from models_LoRAbasedCL.infobudget_gam_rankrecycle_lora import Learner
    elif name == "infoprob_gam_lora":
        from models_LoRAbasedCL.infoprob_gam_lora import Learner
    elif name == "infoprob_lora":
        from models_LoRAbasedCL.infoprob_gam_lora import Learner
    elif name == "infoprobgamlora":
        from models_LoRAbasedCL.infoprob_gam_lora import Learner
    elif name == "fcam_lora":
        from models_LoRAbasedCL.fcam_lora import Learner
    elif name == "fcamgamlora":
        from models_LoRAbasedCL.fcam_lora import Learner
    elif name == "geolora":
        from models.GeoLoRA import Learner
    elif name == "geolora_slowmerge":
        from models_Cl_merge.GeoLoRA_SlowMerge import Learner
    elif name in {"geolora_uv_efm", "geolora-uv-efm", "geouv_efm"}:
        from models_Cl_merge.GeoLoRA_UV_EFM import Learner
    elif name == "geolora_origin":
        from models_LoRAbasedCL.GeoLoRA_origin import Learner
    elif name == "geo_inc_lora":
        from models_LoRAbasedCL.geo_inc_lora import Learner
    elif name == "geoinclora":
        from models_LoRAbasedCL.geo_inc_lora import Learner
    elif name == "fcam":
        from models_LoRAbasedCL.fcam_lora import Learner


    elif name == "finetune":
        from models_CL.finetune import Learner
    elif name in {"scm_finetune", "scmfinetune", "scm"}:
        from models_Cl_merge.scm_finetune import Learner
    elif name == "LPFT":
        from models_CL.LPFT import Learner
    elif name == "linearprobe":
        from models_CL.linearprobe import Learner
    elif name == "lpft_efm":
        from models_CL.LPFT_EFM import Learner


    elif name == "aper_finetune":
        from models_CL.aper_finetune import Learner
    elif name == "aper_ssf":
        from models_CL.aper_ssf import Learner
    elif name == "aper_vpt":
        from models_CL.aper_vpt import Learner 
    elif name == "aper_adapter":
        from models_CL.aper_adapter import Learner
    elif name == "l2p":
        from models_CL.l2p import Learner
    elif name == "dualprompt":
        from models_CL.dualprompt import Learner
    elif name == "coda_prompt":
        from models_CL.coda_prompt import Learner
        
    


    elif name == "icarl":
        from models_CL.icarl import Learner
    elif name == "der":
        from models_CL.der import Learner
    elif name == "coil":
        from models_CL.coil import Learner
    elif name == "foster":
        from models_CL.foster import Learner
    elif name == "memo":
        from models_CL.memo import Learner
    elif name == 'ranpac':
        from models_CL.ranpac import Learner
    elif name == "ease":
        from models_CL.ease import Learner
    elif name == 'slca':
        from models_CL.slca import Learner
    elif name == 'lae':
        from models_CL.lae import Learner
    elif name == 'fecam':
        from models_CL.fecam import Learner
    elif name == 'dgr':
        from models_CL.dgr import Learner
    elif name == 'mos':
        from models_CL.mos import Learner
    elif name == 'cofima':
        from models_CL.cofima import Learner
    elif name == 'duct':
        from models_CL.duct import Learner
    elif name == 'tuna':
        from models_CL.tuna import Learner
    elif name == 'tuna_efm':
        from models_EFM.tuna_efm import Learner

    else:
        assert 0
    return Learner(args)
