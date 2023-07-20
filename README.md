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
