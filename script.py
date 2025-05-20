import os
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import webbrowser

def carregar_imagens_da_pasta(pasta):
    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    imagens = []
    nomes = []
    for nome in arquivos:
        caminho = os.path.join(pasta, nome)
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # redimensiona para tamanho fixo
            imagens.append(img.flatten())
            nomes.append(caminho)  # adiciona o caminho da imagem
        else:
            print(f"Falha ao carregar imagem: {caminho}")
    return np.array(imagens), nomes

# Pastas com imagens (ajuste os caminhos se necessário)
pasta_covid = 'radiografia/COVID'
pasta_normal = 'radiografia/Normal'
pasta_viral = 'radiografia/Viral Pneumonia'

# Carregar todas as imagens de cada pasta
covid_imgs, nomes_covid = carregar_imagens_da_pasta(pasta_covid)
normal_imgs, nomes_normal = carregar_imagens_da_pasta(pasta_normal)
viral_imgs, nomes_viral = carregar_imagens_da_pasta(pasta_viral)

print(f"COVID: {covid_imgs.shape[0]} imagens carregadas")
print(f"Normal: {normal_imgs.shape[0]} imagens carregadas")
print(f"Viral Pneumonia: {viral_imgs.shape[0]} imagens carregadas")

# Verificar se as imagens foram carregadas
if covid_imgs.size == 0 or normal_imgs.size == 0 or viral_imgs.size == 0:
    raise ValueError("Alguma classe não tem imagens carregadas. Verifique as pastas e arquivos.")

# Criar dataset e labels
X = np.vstack([covid_imgs, normal_imgs, viral_imgs])
y = np.array([0]*len(covid_imgs) + [1]*len(normal_imgs) + [2]*len(viral_imgs))
labels = ['COVID', 'Normal', 'Viral Pneumonia']
nomes_todas = nomes_covid + nomes_normal + nomes_viral

# Normalizar features
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# Aplicar PCA com 3 componentes
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X_normalizado)

print("Variância explicada pelos componentes:", pca.explained_variance_ratio_)

# Criar DataFrame para visualização
df = pd.DataFrame({
    'PC1': X_pca_3d[:,0],
    'PC2': X_pca_3d[:,1],
    'PC3': X_pca_3d[:,2],
    'Classe': [labels[i] for i in y],
    'Imagem': nomes_todas  # adiciona os nomes das imagens
})

# Criar gráfico 3D interativo com Plotly
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Classe',
                    hover_name='Imagem',
                    title='PCA 3D - Tomografias pulmonares',
                    width=900, height=700)

# Salvar gráfico em HTML e abrir automaticamente no navegador
arquivo_html = "pca_3d_plot.html"
fig.write_html(arquivo_html)
print(f"Gráfico salvo em {arquivo_html}. Abrindo no navegador...")
webbrowser.open(arquivo_html)