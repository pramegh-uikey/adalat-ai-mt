"""adalat_mt — English→Hindi legal translation for Indian court judgments.

Subpackages:
    data          cleaning, sentence segmentation, alignment and document-level splits
    tokenization  tokenizer efficiency study and vocabulary extension
    inference     batched translation with the MT systems
    training      LoRA adaptation
    evaluation    BLEU/chrF/COMET, bootstrap significance, token usage
    reporting     tables and figures generated from results/
"""

__version__ = "0.1.0"
