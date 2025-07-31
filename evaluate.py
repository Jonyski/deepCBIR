import logging
from app.models import deepCBIR

# Configura um logger básico para ver as mensagens do modelo durante a inicialização.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    print("Inicializando o sistema CBIR para avaliação")
    cbir_system = deepCBIR()
    retrieval_scope = 10

    print(f"\nIniciando a avaliação com um scope (k) de {retrieval_scope}")
    
    cbir_system.evaluate_precision_recall(scope=retrieval_scope)

    print("\nAvaliação concluída.")