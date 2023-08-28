# korean-LLM-quantize
autoGPTQ를 활용한 koalpaca &amp; kullm 모델 양자화 및 테스트

## 실행

#### KoAlpaca
```bash
python quant_with_LLM.py --pretrained_model_dir beomi/KoAlpaca-Polyglot-12.8B --quantized_model_dir ./model/koalpaca-8bit
```

#### Kullm
```bash
python quant_with_LLM.py --pretrained_model_dir nlpai-lab/kullm-polyglot-12.8b-v2 --quantized_model_dir ./model/kullm-8bit
```

### How to use GPTQ model
```python
import torch
from transformers import pipeline
from auto_gptq import AutoGPTQForCausalLM

from utils.prompter import Prompter

MODEL = "j5ng/kullm-12.8b-GPTQ-8bit"
model = AutoGPTQForCausalLM.from_quantized(MODEL, device="cuda:1", use_triton=False)

pipe = pipeline('text-generation', model=model,tokenizer=MODEL)

prompter = Prompter("kullm")

def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(
        prompt, max_length=512,
        temperature=0.2,
        repetition_penalty=3.0,
        num_beams=5,
        eos_token_id=2
    )
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result

instruction = """
손흥민(한국 한자: 孫興慜, 1992년 7월 8일 ~ )은 대한민국의 축구 선수로 현재 잉글랜드 프리미어리그 토트넘 홋스퍼에서 윙어로 활약하고 있다.
또한 대한민국 축구 국가대표팀의 주장이자 2018년 아시안 게임 금메달리스트이며 영국에서는 애칭인 "쏘니"(Sonny)로 불린다.
아시아 선수로서는 역대 최초로 프리미어리그 공식 베스트 일레븐과 아시아 선수 최초의 프리미어리그 득점왕은 물론 FIFA 푸스카스상까지 휩쓸었고 2022년에는 축구 선수로는 최초로 체육훈장 청룡장 수훈자가 되었다.
손흥민은 현재 리그 100호를 넣어서 화제가 되고 있다.
"""
result = infer(instruction=instruction, input_text="손흥민의 애칭은 뭐야?")
print(result) # 손흥민의 애칭은 "쏘니"입니다.
```

### Reference

[EleutherAI/polyglot](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)
[고려대학교/kullm](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2)
[GPTQ](https://github.com/IST-DASLab/gptq)

