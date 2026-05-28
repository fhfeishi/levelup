# RAG retrieve-augmentation-generation

chunking is architecture of RAG
window-chunking   semantic-chunking
chunking-varible: 
- chunk_size,  [200-1000]    # maybe no use in production (never use!); just for test or toy.
- chunk_overlap, [10%-20%]   #  \n\n  \n  .  , ' '  paragraphs -> lines -> sentenses -> clauses -> words -> characters
- split_boundaries [fixed, recursive, semantic], 
- context_type [code legal markdown]  -- each needs different treatment
---  chunking in doc-loading ---
---  chunking after doc-loading ---
late chunking 

``` 
contend-type   strategy      chunk-size
general-docs   recursive     500-1000
technical      semantic       Auto
code           code-spliter  function
markdown       md-spliter     headers

recursive-chunking: 80%
Semantic-chunking: 20%       [may slow and expensive]

start with recursive-chunking, then use semantic-chunking when needed.
```






