# Agenda

1. Embedding model: BERT \-\> SBERT \-\> MTEB \-\> BGE \-\> NV-Embed \-\> Qwen3-Embedding  
2. From RAG to Agentic RAG: Native RAG, ReAct RAG, Agentic RAG. Query Rewrite, Multi-Query, Adaptive RAG, Graph RAG, RAPTOR, Reranker, Self-correction, corrective RAG, FLARE

# Embedding Model: BERT

Basic introduction: Encoder only Transformer, Bidirectional, Masked Language Model MLM, 

# Embedding Model: BERT

How pretraining is done

1. Bidirectional MLM  
2. Next Sentence Prediction NSP

# Embedding Model: BERT

How fine-tuning is done: labeled data, including Question-Answer Pair

# Embedding Model: SBERT

Why BERT has poor performance in semantic text similarity (sentence level).   
Cross Encoder vs Bi-Encoder

# Embedding Model: SBERT

Siamese Network, training loss

# Embedding Model: SBERT

Performance

# Embedding Model: MTEB

Introduction of MTEB benchmark

# Embedding Model: BGE

Chinese MTEB

# Embedding Model: BGE

Training 3 stages: 1\. RetroMAE

# Embedding Model: BGE

Training 3 stages: 2\. Contrastive Loss 

# Embedding Model: BGE

Training 3 stages: 3\. Fine-tuning  
Performance

# Embedding Model: Nv-Embed

Introduction: Decoder-only  
Latent attention layer  
2-stage training while second stage doesn’t use in-batch negative

# Embedding Model: Qwen3-Embedding

Introduction

# Embedding Model: Qwen3-Embedding

Embedding model: Input: {Inst}{Q}\<EOS\> \-\> EOS Pooling  
Reranker model: {Inst}{Q}{Doc} \-\> LM Head \-\> “Yes””No” token \-\> score softmax  
Training loss: InfoNCE style contrastive loss

# Agentic RAG: Native RAG weakness

Chunk size affects paragraph information  
Semantic search directly affects doc

# Agentic RAG: Summary

Summary graph of directions in agentic RAG, including query side rewrite or multi-query, retrieval side, knowledge base side like graphRAG, post-retrieval side like self-reflection or corrective RAG, other agents like Adaptive RAG

# Backup

FLARE:  
利用LLM本身地confidence score判断是不是需要retrieval  
Direct：下一句生成先看下confidence score （prob），然后直接拿这句话去搜索  
Instruct：根据当前地句子，用一个单独的LLM去生成用来search的query，然后Retrieval

SELF-RAG:  
训练一个模型来critiqueLLM output，触发Retrieval trigger