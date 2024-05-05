import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
with torch.autocast("cuda"): 
    device = "cuda"
    
def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data


def main(
    load_8bit: bool = False,
    base_model: str = "",
    # the infer data, if not exists, infer the default instructions in code
    instruct_dir: str = "",
    use_lora: bool = True,
    lora_weights: str = "tloen/alpaca-lora-7b",
    # The prompt template to use, will default to med_template.
    prompt_template: str = "med_template",
):
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=1024,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    def infer_from_json(instruct_dir):
        input_data = load_instruction(instruct_dir)
        for d in input_data:
            instruction = d["instruction"]
            output = d["output"]
            print("###infering###")
            model_output = evaluate(instruction)
            print("###instruction###")
            print(instruction)
            print("###golden output###")
            print(output)
            print("###model output###")
            print(model_output)

    if instruct_dir != "":
        infer_from_json(instruct_dir)
    else:
        for instruction in [
            "左肺癌治疗后复查： 左下肺近肺门旁团块影，较前稍缩小； 左侧肺不张、两肺膨胀不全； 纵隔及肺门淋巴结稍大，考虑转移，较前缩小； 两侧胸腔、心包积液，较前增多； 多发骨转移瘤，较前增多、增大； 肝S2、6血管瘤；肝S3囊肿；肾囊肿。 两侧肾下腺增粗，较前相仿，建议随诊。肺癌治疗后复查，与2014-12-1片对比： 两肺透亮度减低，左肺体积缩小。左侧支气管变窄，左肺下叶近肺门旁见斑片影，大小约31mm×29mm，边界不清，密度不均，边缘见支气管气相，增强扫描可见不均匀较明显强化。左肺中叶近肺门区纵隔旁见楔形稍致密影，其内见支气管气相。余两肺纹理增粗模糊，未见明显结节及肿块。 气管及右侧支气管分枝通畅。左上气管旁、主肺动脉窗、隆突上及左肺门见多发淋巴结，大者短径约8mm，边界欠清，增强扫描轻度强化。心包腔及两侧胸腔积液，较前增多。两侧胸膜未见增厚。 肝脏形态、比例未见明显异常，轮廓光整，肝S2前缘、S6内缘各见一斑片状稍低密度灶，大者约8mm×15mm，边界清晰，增强扫描动脉期结节样强化，门脉期强化范围扩大。肝S3见一囊状低密度灶，直径约5mm，界清，未见强化。 肝内外胆管未见扩张，胆囊不大，未见结石影。胆总管未见扩张。 肝门区未见明确病变。门静脉未见异常。  脾不大，密度均匀。胰腺大小、形态未见异常，密度均匀。  两肾可见小囊状低密度灶，直径约2mm~6mm，界清，无强化。两侧肾下腺增粗，最厚约6mm，增强扫描均匀强化。 腹膜后未见肿大淋巴结。  各胸椎、所示腰椎、右侧肱骨头、胸骨、双侧肋骨见多发斑片及结节状致密灶，边界较清，较前增多、范围增大。",
            "左上肺癌综合治疗后，对比2016-07-08片：左枕叶、左颞叶及第3脑室周围白质区病灶，考虑转移瘤，大致同前。两侧额顶叶小缺血灶同前。空蝶鞍。左上肺癌脑转移综合治疗后，对比2016-07-08片：左枕叶、左颞叶及第3脑室周围白质区见数个异常信号灶，较大者约24mm×16mm（左侧枕叶），边界欠清，T1WI呈稍等信号，T2WI及Flair呈稍高信号，DWI呈稍高信号，增强后呈明显强化，较前未见明显变化；左侧枕叶及颞叶病灶旁软脑膜不均匀增厚，明显强化，较前未见明显变化。两侧额顶叶白质内见几个斑点状、小斑片状异常信号影，T1WI呈等或稍低信号，T2WI及FLAIR下呈高信号，增强后未见强化，较前未见明显变化。小脑及脑干形态正常，信号均匀，增强后未见异常信号区。各脑室及脑池大小、形态未见异常。大脑中线结构未见移位，未见占位性病变。颅骨骨质未见破坏。蝶鞍内见脑脊液充填，垂体变薄，较前未见明显变化。扫描范围内所见两侧下颌窦、筛窦、蝶窦充气良好，骨壁完整。两眼球后未见异常信号。",
            "肺癌肝转移化疗后： 左肺下叶前段、中叶、前纵隔肿块，较前缩小。 左肺下叶、中叶条片状病变，考虑阻塞性肺炎，较前好转。 肝S4转移瘤较前缩小。 肝内多发囊肿。 肝S7肝内胆管小结石。 右肾囊肿。 脾稍大。肺癌肝转移化疗后，与2014-11-18片对比： 左肺下叶前段、中叶、前纵隔见一肿块，大小约67mm×96mm×113mm，边界欠清，边缘不光整，呈分叶状，内见斑点状钙化，增强扫描呈不均匀明显强化，内见片状无强化坏死区，与前片对比，肿块较前缩小，肿块部分包绕升主动脉，压迫下腔静脉、左肺动脉、左下肺静脉，左肺下叶、中叶见片状密度增高影，边界模糊，较前好转。右肺纹理清晰，肺内未见实质性病变。  左肺下叶、中叶支气管受压、变窄。 左肺门见一肿大淋巴结，大小约7mm×9mm，增强扫描见强化，较前缩小。右肺门未见明显肿大淋巴结。两侧胸腔未见积液。两侧胸膜未见增厚、粘连。 肝脏形态、各叶比例未见异常，其外形轮廓光整，肝S4见2个结节，较大者约7mm×7mm，边界不清，增强扫描呈轻度强化，较前缩小。肝S7见一点状致密灶，直径约2mm。肝内见多个类圆形低密度灶，最大者约34mm×25mm，边界清晰，增强扫描未见强化。肝内外胆管未见异常，其内未见结石，胆囊未见异常。肝门区未见异常。门静脉未见异常。  脾脏稍大，密度未见异常。 胰腺大小、形态、密度未见异常，增强扫描未见异常强化。 右肾见数个类圆形低密度灶，最大者约25mm×24mm，边界清晰，增强扫描未见强化。左肾及两肾下腺未见异常。  腹主动脉旁未见肿大淋巴结。 扫描范围内所见骨质未见明确破坏征象。",
            "左上肺癌综合治疗后，对比2016-05-16片：左枕叶、左颞叶及第3脑室周围白质区病灶，考虑转移瘤，范围较前稍增大。两侧额顶叶小缺血灶同前。空蝶鞍。左上肺癌综合治疗后，对比2016-05-16片：左枕叶、左颞叶及第3脑室周围白质区可见数个异常信号灶，较大者约24mm×17mm（左侧枕叶），边界欠清，T1WI呈稍等信号，T2WI及Flair呈稍高信号，DWI呈稍高信号，增强后呈明显强化，病灶范围较前稍增大；左侧枕叶及颞叶病灶旁软脑膜不均匀增厚，明显强化，较前明显。两侧额顶叶白质内见几个斑点状、小斑片状异常信号影，T1WI呈等或稍低信号，T2WI及FLAIR下呈高信号，增强后未见强化，较前未见明显变化。小脑及脑干形态正常，信号均匀，增强后未见异常信号区。各脑室及脑池大小、形态未见异常。大脑中线结构未见移位，未见占位性病变。颅骨骨质未见破坏。蝶鞍内见脑脊液充填，垂体变薄，较前未见明显变化。扫描范围内所见两侧下颌窦、筛窦、蝶窦充气良好，骨壁完整。两眼球后未见异常信号。",
        ]:
            print("Instruction:", instruction)
            print("Response:", evaluate(instruction))
            print()


if __name__ == "__main__":
    fire.Fire(main)
