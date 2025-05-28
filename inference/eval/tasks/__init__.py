
# task files
# from . import agi_eval
# from . import aqua
# from . import arc
# from . import boolq
from . import c_eval
# from . import c3
# from . import cmrc2018
# from . import commonsenseqa
# from . import coqa
from . import gsm8k
# from . import hellaswag
from . import humaneval
from . import mmlu
# from . import openbookqa
# from . import piqa
# from . import race
# from . import triviaqa
# from . import winogrande
# from . import wsc273

# from . import custom_prompt


########################################
# All tasks
########################################

TASK_REGISTRY = {
    # commonsense
    # "boolq": boolq.BoolQ,
    # "piqa": piqa.PIQA,
    # "hellaswag": hellaswag.HellaSwag,
    # "winogrande": winogrande.Winogrande,
    # "wsc273": wsc273.WinogradSchemaChallenge273,
    # "arc_easy": arc.ARCEasy,
    # "arc_challenge": arc.ARCChallenge,
    # "openbookqa": openbookqa.OpenBookQA,
    
    # knowledge
    # "trivia_qa": triviaqa.TriviaQA_gen,

    # reading comprehension
    # "coqa": coqa.CoQA,
    # "race_middle": race.RACE_Middle,
    # "race_high":race.RACE_High,
    # "cmrc2018": cmrc2018.CMRC2018,
    # "c3_dialog": c3.C3_dialog,
    # "c3_mixed": c3.C3_mixed,

    # math
    "gsm8k": gsm8k.GradeSchoolMath8K,
    "gsm8k_direct": gsm8k.GSM8K_direct,
    # "aqua": aqua.AQuA,
    # "aqua_direct": aqua.AQuA_direct,
    # "commonsenseqa": commonsenseqa.CommonsenseQA,
    # "commonsenseqa_direct": commonsenseqa.CommonsenseQA_direct,

    # code
    "humaneval": humaneval.HumanEval,

    # benchmark
    "mmlu": mmlu.MMLU_fast,
    "mmlu_stem": mmlu.MMLU_stem,
    "mmlu_social": mmlu.MMLU_social,
    "mmlu_human": mmlu.MMLU_human,
    "mmlu_other": mmlu.MMLU_other,
    "mmlu_CoT": mmlu.MMLU_ZeroShot_CoT_ChatGen,
    "c_eval": c_eval.CEval_fast,
    "c_eval_zeroshot": c_eval.CEval_zeroshot,
    "c_eval_stem": c_eval.CEval_stem,
    "c_eval_social": c_eval.CEval_social,
    "c_eval_human": c_eval.CEval_human,
    "c_eval_other": c_eval.CEval_other,
    "c_eval_CoT": c_eval.CEval_CoT_Gen,
    # "agi_gaokao_multichoice": agi_eval.GaoKao_multichoice_prob,
    # "agi_gaokao_multichoice_gen": agi_eval.GaoKao_multichoice_gen,
    # "agi_gaokao_multichoice_CoT_gen": agi_eval.GaoKao_multichoice_CoT_gen,
    # "agi_gaokao_multichoice_zeroCoT_chat": agi_eval.GaoKao_multichoice_zeroCoT_chat,
    # "agi_gaokao_cloze": agi_eval.GaoKao_cloze,
    # "agi_gaokao_cloze_direct": agi_eval.GaoKao_cloze_direct,

    # other
    # "custom_prompt_zh2": custom_prompt.CustomPromptZH,
    # "custom_prompt_en": custom_prompt.CustomPromptEN,
    # "knowledge_prompt": custom_prompt.CustomPromptZH_v1

}

ALL_TASKS = TASK_REGISTRY.keys()



# TODO bbh
# TODO chid
# TODO drop
# TODO arithmetic
# TODO in-context learning 