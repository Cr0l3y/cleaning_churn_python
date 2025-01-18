# Limpeza e tratamento de dados


*Objetivo: Tratamento na base de dados para retirar dados nulos, duplicatas, feature ou colunas categoricas. Deixando a base de dados
pronta para ser adicionada em um modelo de machine learning.*


Algumas colunas estavam em um formato aninhado, o que dificultava a análise. Para resolver isso, utilizamos o método json_normalize(), e transformando e um formato mais adequado, assim criando novas colunas a partir das chaves do JSON.

Foi usado dataset-telecon.json ultilizando o método with(), que garante o fechamento do arquivo após o uso.

Visualizamos as primeiras cinco linhas do DataFrame normalizado, onde pudemos observar como as colunas foram estruturadas a partir do JSON original.

