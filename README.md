## seed == 1234
## FLAN-T5-BASE
### Baseline
- bleu-2: 0.2848365152512618
- rouge-L: 0.0615727811931462
- perplexity: 166.62268956543832
### Lora (input == 256 && output == 32)
- bleu-2: 0.16733235972058488
- rouge-L: 0.12587111501114437
- perplexity: 54.45750042576675
### Lora (input == 256 && output == 64)
- bleu-2: 0.19251442154829138
- rouge-L: 0.08709721298735355
- perplexity: 54.038574459765734
### Lora (input == 512 && output == 32)
- bleu-2: 0.02830188679245283
- bleu-4: 0.005208333333333333
- rouge-L: 0.1526036770097509
- perplexity: 16.96456595361225
### Lora (input == 512 && output == 64)
