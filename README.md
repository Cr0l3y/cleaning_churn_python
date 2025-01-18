EN
# Data cleaning and processing

*Objective: Processing the database to remove null data, duplicates, features or categorical columns. Leaving the database ready to be added to a machine learning model.*

Some columns were in a nested format, which made analysis difficult. To solve this, we used the json_normalize() method, and transformed it into a more appropriate format, thus creating new columns from the JSON keys.

Dataset-telecon.json was used using the with() method, which ensures that the file is closed after use.

We visualized the first five lines of the normalized DataFrame, where we could see how the columns were structured from the original JSON.




PT-BR


# Limpeza e tratamento de dados


*Objetivo: Tratamento na base de dados para retirar dados nulos, duplicatas, feature ou colunas categoricas. Deixando a base de dados
pronta para ser adicionada em um modelo de machine learning.*


Algumas colunas estavam em um formato aninhado, o que dificultava a análise. Para resolver isso, utilizamos o método json_normalize(), e transformando e um formato mais adequado, assim criando novas colunas a partir das chaves do JSON.

Foi usado dataset-telecon.json ultilizando o método with(), que garante o fechamento do arquivo após o uso.

Visualizamos as primeiras cinco linhas do DataFrame normalizado, onde pudemos observar como as colunas foram estruturadas a partir do JSON original.








