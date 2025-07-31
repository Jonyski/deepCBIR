from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
import os
import logging

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# É uma boa prática ter um logger configurado.
logger = logging.getLogger(__name__)

class deepCBIR:
    def __init__(self):
        logger.info("App is loading. It will load the model and also vectorize the database image. Depending on your"\
                    "resource it may take upto several minutes")
        logger.info("Loading CBIR Model")
        self.load_cbir_model()

        logger.info("Vectorizing Database")
        self.vectorize_database("./app/database")

    def load_cbir_model(self):
        self.cbir_model = InceptionResNetV2(weights="imagenet", include_top=True, input_shape=(299, 299, 3))
        self.cbir_model = Model(inputs=self.cbir_model.input, outputs=self.cbir_model.get_layer("avg_pool").output)


    def vectorize_database(self, database_dir):
        try:
            self.database = np.load("./app/static/database.npy", allow_pickle=True).item()
        except:
            img_paths = glob.glob(database_dir+"/*")
            self.database = {}
            for img_url in tqdm(img_paths):
                try:
                    self.database[img_url] = self.img_to_encoding(img_url, self.cbir_model)
                except:
                    pass

        self.features = np.array(list(self.database.values())).reshape(len(self.database), -1)

    def img_to_encoding(self, image_path, model):
        img1 = image.load_img(image_path, target_size=(299, 299, 3))
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        embedding = model.predict_on_batch(x)
        return embedding

    def retrieve_images(self, query_img_path, scope, debug):
        query = self.img_to_encoding(query_img_path, self.cbir_model)

        # MODO ANTIGO DE ANÁLISE DE SIMILARIDADE (VIA DISTÂNCIA EUCLIDIANA)
        #   dist_vec = np.linalg.norm(self.features - query, axis=1)
        #   df = pd.DataFrame({"image_url":list(self.database.keys()), "distance":dist_vec})
        #   df = df.sort_values("distance").reset_index(drop=True)
        #   return df["image_url"][:scope].tolist()

        # MODO NOVO DE ANÁLISE DE SIMILARIDADE (VIA ELO)
        sum_diff = np.abs(self.features.sum(axis=1) - query.sum())
        chebyshev_dist = np.max(np.abs(self.features - query), axis=1)
        sorted_indices = np.lexsort((chebyshev_dist, sum_diff))
        image_urls = np.array(list(self.database.keys()))
        sorted_urls = image_urls[sorted_indices]

        # DEBUG -----------------------------------------------------------
        if debug:
            # Pega os valores das chaves na ordem em que foram classificados
            sorted_sum_diff = sum_diff[sorted_indices]
            sorted_chebyshev = chebyshev_dist[sorted_indices]
            
            print(f"Soma do Vetor de Consulta: {query.sum():.4f}\n")
            
            # Imprime o cabeçalho da tabela de resultados
            header = f"{'Rank':<5} | {'Image URL':<30} | {'Chave Primária (Sum Diff)':<30} | {'Chave Secundária (Chebyshev)'}"
            print(header)
            print("-" * len(header))

            # Itera sobre os n primeiros resultados para imprimir seus detalhes
            for i in range(scope):
                rank = i + 1
                url = sorted_urls[i]
                primary_key = sorted_sum_diff[i]
                secondary_key = sorted_chebyshev[i]
                info = f"{rank:<5} | {url:<30} | {primary_key:<30.4f} | {secondary_key:.4f}"
                print(info)
            
            print("-" * len(header))
        # FIM DO DEBUG ----------------------------------------------------
        
        return sorted_urls[:scope].tolist()

    def evaluate_precision_recall(self, scope):
        # Construindo o mapa com a categoria de cada imagem: { 'nome_do_arquivo.jpg': 'categoria', ... }
        category_map = {}
        category_paths = glob.glob(os.path.join("./app/database/101_ObjectCategories", "*"))
        for category_path in category_paths:
            if not os.path.isdir(category_path): continue
            category_name = os.path.basename(category_path)
            image_files = glob.glob(os.path.join(category_path, "*"))
            for img_path in image_files:
                img_filename = os.path.basename(img_path)
                category_map[img_filename] = category_name

        # Contando o total de imagens por categoria para o cálculo da revocação
        category_counts = {}
        for category in category_map.values():
            category_counts[category] = category_counts.get(category, 0) + 1

        # Executando a avaliação
        all_images = list(self.database.keys())
        num_images_evaluated = 0
        total_precision = 0
        total_recall = 0

        for query_path in tqdm(all_images, desc="Evaluating Precision & Recall"):
            query_filename = os.path.basename(query_path)

            # Pulando arquivos que não têm uma categoria definida
            if query_filename not in category_map:
                continue

            num_images_evaluated += 1
            query_category = category_map[query_filename]
            
            # Obtendo os resultados da busca.
            retrieved_paths = self.retrieve_images(query_path, scope=scope, debug=False)
            
            # Contando os acertos.
            true_positives = 0
            for path in retrieved_paths:
                retrieved_filename = os.path.basename(path)
                if category_map.get(retrieved_filename) == query_category:
                    true_positives += 1
            
            # Calcula precisão e revocação para esta consulta.
            precision = true_positives / scope if scope > 0 else 0
            total_relevant_in_category = category_counts.get(query_category, 0)
            recall = true_positives / total_relevant_in_category if total_relevant_in_category > 0 else 0
            
            total_precision += precision
            total_recall += recall

        # Calculando as médias e exibindo os resultados.
        mean_precision = (total_precision / num_images_evaluated) * 100 if num_images_evaluated > 0 else 0
        mean_recall = (total_recall / num_images_evaluated) * 100 if num_images_evaluated > 0 else 0
        
        print("\n____ Resultados da Avaliação ____")
        print(f"Scope (k) .............: {scope} imagens recuperadas por consulta")
        print(f"Total de Consultas ....: {num_images_evaluated} imagens avaliadas")
        print(f"Precisão Média ........: {mean_precision:.2f}%")
        print(f"Revocação Média .......: {mean_recall:.2f}%")

    def create_plot(self, image_paths):
        pass

        if len(image_paths) == 1:
            img = Image.open(image_paths[0])
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_title('Query Image')
            ax.axis("off")
            fig.savefig("./app/tmp/query.jpg")
        else:
            rows = (len(image_paths) // 5)
            if len(image_paths) % 5 != 0:
                rows += 1
            if rows == 1:
                cols = len(image_paths)
            else:
                cols = 5
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5))
            fig.suptitle('Retrieved Images', fontsize=20)
            for i in range(len(image_paths)):
                x = i % 5
                y = i // 5
                img = Image.open(image_paths[i])
                img = img.resize((299, 299))
                if rows == 1:
                    axes[x].imshow(img)
                    axes[x].axis("off")
                else:
                    axes[y, x].imshow(img)
                    axes[y, x].axis("off")

            fig.savefig("./app/tmp/retrieved.jpg")