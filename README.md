# Extrator de rostos e recorte

O programa tem como objetivo receber um dataset de pessoas e extrair apenas as potenciais faces contidas em cada imagem. Para realizar a detecção, uma rede Caffe ```res10_300x300_ssd_iter_140000.caffemodel``` pré treinada é utilizada, junto com seu modelo ```deploy.prototxt``` (Para maiores informaçoes sobre a rede pré treinada, veja a sessão REFERENCAS). Após fazer a detecção dos rostos em cada imagem (1° filtragem), o programa os recorta e envia para uma pasta determinada pelo usuário. Quando o processo for finalizado, a rede é utilizada novamente (2° filtragem) para verificar se os recortes realmente são faces, e eliminar os falsos positivos. Além disso, ao final de cada etapa, um arquivo .txt é gerado contendo informaçoes sobre as imagens como seu nome, tamanho e probabilidade de ser um rosto. Às etapas principais do programa são descritas abaixo:

## Primeira filtragem (raw)

Nessa etapa, o programa ```detect_faces.py``` passa por cada imagem do dataset e faz a primeira detecção das n-faces de cada uma e, baseando-se no seu bounding box, efetua os n-recortes na imagem e os envia para uma pasta determinada, que por padrão é ```./first-filter/multi-crop-raw```. Vale ressaltar que aumentei o tamanho do bounding box para o recorte abrenger o rosto por completo. Confira em **detect_faces.py, linha 90**:

```python
startX, startY, endX, endY = startX - 2, startY - 10, endX + 2, endY + 10
``` 

O programa também salva uma cópia de cada imagem original com os bounding boxes ***desenhados***, assim da pra ter um resultado mais "visual" das detecções da rede. Por padrão a pasta de destino é  ```./first-filter/bboxes```

É importante salientar que nessa etapa, o limiar da probabilidade de ser um rosto, confidence, deve ser baixo, logo tudo que possui uma probabilidade acima do valor do confidence será interpretado como rosto (útil para imagens com varias pessoas ou com qualidade baixa). Portanto, a rede irá detectar o máximo possível de faces da imagem, independente do tamanho e luminosidade.

A pasta **first-filter** por padrão contem todas as informaçoes uteis da primira filtragem. Sua estrutura interna é a seguinte:

<p align="center">
  <img src="./screenshots/first-filter-folder.jpg" alt="Estrutura da pasta first-filter">
</p>

A pasta **bboxes** contem todas as imagens em que ao menos um rosto foi encontrado, com o bounding box desenhado em cada "rosto" da imagem, junto com o confidence. Nessa pasta, fica claro como falsos positivos são frequentes na primeira filtragem. Abaixo é mostrado como são salvos os arquivos da pasta e um exemplo de uma imagem com um falso positivo.

<p float="left">
  <img src="./screenshots/bbox-folder.jpg" width="100" alt="Pasta bboxes"/>
  <img src="./screenshots/bboxes.jpg" width="100" alt="Imagem com seus bounding boxes desenhados"/> 
</p> 

A pasta **multi-crop-raw** contém efetivamente todos os recortes dos mesmos bounding boxes mostrados na pasta **bboxes**

<p float="left">
  <img src="./screenshots/multi-raw-folder.jpg" width="100" alt="Pasta multi-crop-raw"/>
  <img src="./screenshots/raw.jpg" width="100" alt="Recorte"/> 
</p> 

A pasta **not** contém imagens que nenhum rosto foi detectado, é util ter essa pasta pois podemos verificar se existem falsos negativos e realoca-los.

## Segunda filtragem 

Como na primeira filtragem o confidence foi baixo, praticamente todos os rostos foram encontrados, entretando isso causa um problema: muitos falsos positivos (imagens que não eram um rosto mas a rede detectou como sendo um) também são filtrados. Isso gera um maior volume de recortes em que muitos deles não condizem com o resultado esperado (apenas faces). Abaixo um exemplo de falso positivo recortado na primeira fase:


Porém, nessa fase, como os as imagens (recortes das originais) estão isoladas, o programa consegue destinguir com maior facilidade se é um rosto ou não, e separada as imagens selecinadas e não selecionadas em suas devidas pastas, além de gerar um arquivo .txt contendo informações como nome, confidence e tamanho de cada imagem