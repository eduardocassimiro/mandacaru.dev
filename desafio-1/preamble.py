import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import json
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from keras.layers import BatchNormalization
plt.style.use('ggplot')

######################## Notebook 1 #################################
# https://github.com/eduardocassimiro/mandacaru.dev/blob/main/desafio-1/testes-de-hipotese/Teste%20de%20Hip%C3%B3tese%201%20-%20Verificando%20os%20Canais%20de%20Cores%20de%20Diferentes%20Tipos%20de%20Documentos.ipynb

# Abrindo os arquivos
def plot_1():
	cpf_frente = cv2.imread('imagens_originais/cpf_frente.jpg')
	cpf_verso = cv2.imread('imagens_originais/cpf_verso.jpg')
	rg_frente = cv2.imread('imagens_originais/rg_frente.jpg')
	rg_verso = cv2.imread('imagens_originais/rg_verso.jpg')
	cnh_frente = cv2.imread('imagens_originais/cnh_frente.jpg')
	cnh_verso = cv2.imread('imagens_originais/cnh_verso.jpg')

	cpf_frente = cv2.cvtColor(cpf_frente, cv2.COLOR_BGR2RGB)
	cpf_verso = cv2.cvtColor(cpf_verso, cv2.COLOR_BGR2RGB)
	rg_frente = cv2.cvtColor(rg_frente, cv2.COLOR_BGR2RGB)
	rg_verso = cv2.cvtColor(rg_verso, cv2.COLOR_BGR2RGB)
	cnh_frente = cv2.cvtColor(cnh_frente, cv2.COLOR_BGR2RGB)
	cnh_verso = cv2.cvtColor(cnh_verso, cv2.COLOR_BGR2RGB)

	fig = plt.figure(figsize=(12,6))

	fig.add_subplot(2,3,1); plt.axis('off')
	plt.title('cpf_frente')
	plt.imshow(cpf_frente)

	fig.add_subplot(2,3,2); plt.axis('off')
	plt.title('cpf_verso')
	plt.imshow(cpf_verso)

	fig.add_subplot(2,3,3); plt.axis('off')
	plt.title('rg_frente')
	plt.imshow(rg_frente)

	fig.add_subplot(2,3,4); plt.axis('off')
	plt.title('rg_verso')
	plt.imshow(rg_verso)

	fig.add_subplot(2,3,5); plt.axis('off')
	plt.title('chn_frente')
	plt.imshow(cnh_frente)

	fig.add_subplot(2,3,6); plt.axis('off')
	plt.title('chn_verso')
	plt.imshow(cnh_verso)

	fig.suptitle('Amostras dos documentos', fontsize=20)
	plt.show();

def plot_2():
	cpf_frente = cv2.imread('imagens_originais/cpf_frente.jpg')
	cpf_verso = cv2.imread('imagens_originais/cpf_verso.jpg')
	rg_frente = cv2.imread('imagens_originais/rg_frente.jpg')
	rg_verso = cv2.imread('imagens_originais/rg_verso.jpg')
	cnh_frente = cv2.imread('imagens_originais/cnh_frente.jpg')
	cnh_verso = cv2.imread('imagens_originais/cnh_verso.jpg')

	cpf_frente = cv2.cvtColor(cpf_frente, cv2.COLOR_BGR2RGB)
	cpf_verso = cv2.cvtColor(cpf_verso, cv2.COLOR_BGR2RGB)
	rg_frente = cv2.cvtColor(rg_frente, cv2.COLOR_BGR2RGB)
	rg_verso = cv2.cvtColor(rg_verso, cv2.COLOR_BGR2RGB)
	cnh_frente = cv2.cvtColor(cnh_frente, cv2.COLOR_BGR2RGB)
	cnh_verso = cv2.cvtColor(cnh_verso, cv2.COLOR_BGR2RGB)

	hist_red = cv2.calcHist(cpf_verso,[0],None,[256],[0,256])
	hist_green = cv2.calcHist(cpf_verso,[1],None,[256],[0,256])
	hist_blue = cv2.calcHist(cpf_verso,[2],None,[256],[0,256])
	hist = cv2.calcHist(cpf_verso,[0],None,[256],[0,256])	

	fig = plt.figure(figsize=(12,6))

	fig.add_subplot(2,2,1); plt.axis('off')
	plt.title('cpf_frente')
	plt.imshow(cpf_frente)

	fig.add_subplot(2,2,2);
	plt.title('Canal Vermelho')
	plt.plot(hist_red,color='red',lw=2)

	fig.add_subplot(2,2,3);
	plt.title('Canal Verde')
	plt.plot(hist_green,color='green',lw=2)

	fig.add_subplot(2,2,4);
	plt.title('Canal Azul')
	plt.plot(hist_blue,color='blue',lw=2)

	fig.suptitle('Canais de cores da imagem', fontsize=20)
	plt.show();

##########################################################################

###################### Notebook 1.4 ######################################
# link: https://github.com/eduardocassimiro/mandacaru.dev/blob/main/desafio-1/testes-de-hipotese/Teste%20de%20Hip%C3%B3tese%201.4%20-%20Visualiza%C3%A7%C3%A3o%20das%20M%C3%A9dias%20dos%20Histogramas%20das%20Amostras%20Ruins%20e%20Quantiza%C3%A7%C3%A3o%20de%20Satura%C3%A7%C3%A3o%20.ipynb

# Abrindo os arquivos
# Lista de nomes para facilitar o acesso as pastas e aos arquivos
def plot_3():
	documentos = ['rg_frente','rg_tras','cnh_frente','cnh_tras']

	# Laço para repetir o processos anteriores de um vez para todos as amostras boas
	medias = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_bons = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,20):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        doc.append(cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2RGB))
	        docs_bons.append(doc[i])
	        
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_red = 0
	    sum_green = 0
	    sum_blue = 0
	    
	    for i in range(len(doc)):
	        sum_red += doc[i][0]
	        sum_green += doc[i][1]
	        sum_blue += doc[i][2]
	        
	    sum_red = sum_red/len(doc)
	    sum_green = sum_green/len(doc)
	    sum_blue = sum_blue/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_red, sum_green, sum_blue]}

	# Laço para repetir o processos anteriores de um vez para todas as amostras ruins
	medias2 = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_ruins = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,7):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        doc.append(cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2RGB))
	        docs_ruins.append(doc[i])
	         
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_red = 0
	    sum_green = 0
	    sum_blue = 0
	    
	    for i in range(len(doc)):
	        sum_red += doc[i][0]
	        sum_green += doc[i][1]
	        sum_blue += doc[i][2]
	        
	    sum_red = sum_red/len(doc)
	    sum_green = sum_green/len(doc)
	    sum_blue = sum_blue/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias2[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_red, sum_green, sum_blue]}

	n = len(medias.keys()) + len(medias2.keys())
	fig , ax = plt.subplots(n, 4, figsize=(18,3*n), gridspec_kw={'width_ratios': [1,3,3,3]})

	cont = 0
	ax.flatten()
	ax = ax.T.flatten()
	for i,doc in zip(range(0,n,2),documentos):
	    ax[0+(i)].imshow(medias[doc]['matrix']); 
	    ax[0+(i)].axis('off'); ax[0+(i)].set_title(doc + ' bom',fontsize=10)

	    ax[n+(i)].plot(medias[doc]['hist'][0], color='red');
	    ax[n*2+(i)].plot(medias[doc]['hist'][1], color='green')
	    ax[n*3+(i)].plot(medias[doc]['hist'][2], color='blue')

	    ax[0+(i+1)].imshow(medias2[doc]['matrix']); 
	    ax[0+(i+1)].axis('off'); ax[0+(i+1)].set_title(doc + ' ruim',fontsize=10)

	    ax[n+(i+1)].plot(medias2[doc]['hist'][0], color='red');
	    ax[n*2+(i+1)].plot(medias2[doc]['hist'][1], color='green')
	    ax[n*3+(i+1)].plot(medias2[doc]['hist'][2], color='blue')
	    
	fig.suptitle('Comparação da Média dos Canais de Cores das Amostras Boas e Ruins', fontsize=20)
	plt.show()

###############################################################################

############################### Notebook 1.1 ##################################
# link: https://github.com/eduardocassimiro/mandacaru.dev/blob/main/desafio-1/testes-de-hipotese/Teste%20de%20Hip%C3%B3tese%201.1%20-%20Tentando%20Melhorar%20o%20Contraste%20de%20Imagens%20Coloridas.ipynb

def clahe_func(img):
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    red = clahe.apply(r)
    green = clahe.apply(g)
    blue = clahe.apply(b)
    return cv2.merge((red, green, blue))

# Abrindo os arquivos
def plot_4():
	rg_ruim = cv2.cvtColor(cv2.imread('imagens_originais/rg_frente_2.jpg'), cv2.COLOR_BGR2RGB)
	rg_ruim_cinza = cv2.cvtColor(rg_ruim, cv2.COLOR_RGB2GRAY)

	rg_bom = cv2.cvtColor(cv2.imread('imagens_originais/rg_frente.jpg'), cv2.COLOR_BGR2RGB)
	rg_bom_cinza = cv2.cvtColor(rg_bom, cv2.COLOR_RGB2GRAY)

	rg_ruim_clahe = clahe_func(rg_ruim)
	rg_clahe_cinza = cv2.cvtColor(rg_ruim_clahe, cv2.COLOR_RGB2GRAY)

	documentos = {
	    'rg_ruim' : {'matriz' : rg_ruim},
	    'rg_ruim_clahe': {'matriz' : rg_ruim_clahe},
	    'rg_bom' : {'matriz' : rg_bom}
	}

	for key, value in documentos.items():
	    documentos[key]['hist_red'] = cv2.calcHist(documentos[key]['matriz'],[0],None,[256],[0,256])
	    documentos[key]['hist_green'] = cv2.calcHist(documentos[key]['matriz'],[1],None,[256],[0,256])
	    documentos[key]['hist_blue'] = cv2.calcHist(documentos[key]['matriz'],[2],None,[256],[0,256])

	n = len(documentos.keys())
	fig , ax = plt.subplots(n, 4, figsize=(18,3*n), gridspec_kw={'width_ratios': [1,3,3,3]})

	cont = 0
	ax.flatten()
	ax = ax.T.flatten()
	for key, value in documentos.items():
	    ax[0+(cont)].imshow(value['matriz']); 
	    ax[0+(cont)].axis('off'); ax[0+(cont)].set_title(key,fontsize=10)

	    ax[n+(cont)].plot(value['hist_red'], color='red');
	    ax[n*2+(cont)].plot(value['hist_green'], color='green')
	    ax[n*3+(cont)].plot(value['hist_blue'], color='blue')
	    cont+=1
	    
	fig.suptitle('Canais de cores de rg_frente', fontsize=20)
	plt.show()

##############################################################################

##################### Notebook 1.2 ###########################################

def increase_brightness_percent(img, percent):
    r, g, b = cv2.split(img)
    return cv2.merge((r*(percent*1.002989), g*(percent*1.005870), b*(percent*1.001140))).astype('uint8')

def plot_histograms(img, name, title):
    hist_red = cv2.calcHist(img,[0],None,[256],[0,256])
    hist_green = cv2.calcHist(img,[1],None,[256],[0,256])
    hist_blue = cv2.calcHist(img,[2],None,[256],[0,256])
    
    fig = plt.figure(figsize=(14,7))

    fig.add_subplot(2,2,1); plt.axis('off')
    plt.title(name)
    plt.imshow(img)

    fig.add_subplot(2,2,2);
    plt.title('Canal Vermelho')
    plt.plot(hist_red,color='red',lw=2)

    fig.add_subplot(2,2,3);
    plt.title('Canal Verde')
    plt.plot(hist_green,color='green',lw=2)

    fig.add_subplot(2,2,4);
    plt.title('Canal Azul')
    plt.plot(hist_blue,color='blue',lw=2)

    fig.suptitle(title, fontsize=20)
    plt.show();

def plot_5():
	rg_ruim = cv2.cvtColor(cv2.imread('imagens_originais/rg_frente_2.jpg'), cv2.COLOR_BGR2RGB)
	rg_ruim_cinza = cv2.cvtColor(rg_ruim, cv2.COLOR_RGB2GRAY)

	rg_bom = cv2.cvtColor(cv2.imread('imagens_originais/rg_frente.jpg'), cv2.COLOR_BGR2RGB)
	rg_bom_cinza = cv2.cvtColor(rg_bom, cv2.COLOR_RGB2GRAY)
	rg_p = increase_brightness_percent(rg_ruim, 1.45)
	plot_histograms(rg_p, 'rg com ganho', 'Canais de cores com Ganho Percentual')

def plot_6():
	rg_ruim = cv2.cvtColor(cv2.imread('imagens_originais/rg_frente_2.jpg'), cv2.COLOR_BGR2RGB)
	rg_ruim_cinza = cv2.cvtColor(rg_ruim, cv2.COLOR_RGB2GRAY)

	rg_bom = cv2.cvtColor(cv2.imread('imagens_originais/rg_frente.jpg'), cv2.COLOR_BGR2RGB)
	rg_bom_cinza = cv2.cvtColor(rg_bom, cv2.COLOR_RGB2GRAY)

	rg_clahe = clahe_func(rg_ruim)
	rg_p = increase_brightness_percent(rg_clahe, 1.1)
	plot_histograms(rg_p, 'rg com ganho', 'Canais de cores com Ganho Percentual Aplicado ao Contraste CLAHE')

################################################################################

########################### Notebook 1.5 #######################################

# Abrindo os arquivos
# Lista de nomes para facilitar o acesso as pastas e aos arquivos
def plot_7():
	documentos = ['rg_frente','rg_tras','cnh_frente','cnh_tras']

	docs_bons = [] # Lista para salvar as imagens 
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    for i in range(0,20):   # Laço para normalizar todos os histogramas do referido documento da vez
	        docs_bons.append(cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2HSV))

	docs_ruins = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    for i in range(0,7):   # Laço para normalizar todos os histogramas do referido documento da vez
	        docs_ruins.append(cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2HSV))

	# Calculando as médias de saturação de cada imagem
	media_sat_bons = []
	media_val_bons = []
	media_sat_ruins = []
	media_val_ruins = []

	for h in docs_bons:
	    _, S, V =  cv2.split(h)
	    media_sat_bons.append(np.mean(S))
	    media_val_bons.append(np.mean(V))
	    
	for h in docs_ruins:
	    _, S, V =  cv2.split(h)
	    media_sat_ruins.append(np.mean(S))
	    media_val_ruins.append(np.mean(V))

	# Calculando a média total de saturação de cada tipo de documento
	media_sat_final_ruins = []
	media_val_final_ruins = []
	media_sat_final_bons = []
	media_val_final_bons = []

	for i in range(len(documentos)):
	    media_sat_final_ruins.append(np.mean(media_sat_ruins[7*i:7*(i+1)])) # [7*i:7*(i+1)] iterar sobre o fatiamento
	    media_sat_final_bons.append(np.mean(media_sat_bons[20*i:20*(i+1)]))
	    
	    media_val_final_ruins.append(np.mean(media_val_ruins[7*i:7*(i+1)]))
	    media_val_final_bons.append(np.mean(media_val_bons[20*i:20*(i+1)]))

# Plot para visualizar a diferenças das médias de saturação de cada tipo de documento

	n = len(documentos) # Número de figuras

	fig = plt.figure(figsize=(16,8))

	for i in range(n): # Laço para automatizar as plotagens
	    fig.add_subplot(2,2,i+1) # Adiciona um novo subplot em cada iteração
	        
	    # Plot das médias de saturação das amostras
	    plt.plot(list(range(20)), media_val_bons[20*i:20*(i+1)], c='darkorange') # [20*i:20*(i+1)] iterar sobre a porção de cada
	    plt.plot(list(range(7)), media_val_ruins[7*i:7*(i+1)], c='mediumorchid')   # tipo de documento
	    
	    # Plot da média final de cada tipo de documento em forma de linha
	    plt.axhline(media_val_final_ruins[i], c='mediumorchid')
	    plt.axhline(media_val_final_bons[i], c='darkorange')
	    
	    plt.title(documentos[i])
	    
	fig.legend(['Amostra Boa','Amostra Ruim'], loc='center right')
	fig.suptitle('Média de "Valor" das Amostras Boas e Ruins (HSV)', fontsize=20)
	plt.show();

################################################################################

######################### Notebook 1.8 #########################################
# link: https://github.com/eduardocassimiro/mandacaru.dev/blob/main/desafio-1/testes-de-hipotese/Teste%20de%20Hip%C3%B3tese%201.8%20-%20Visualiza%C3%A7%C3%A3o%20dos%20Histogramas%20das%20Matizes%20Equalizadas%20e%20M%C3%A9dia%20M%C3%B3vel.ipynb

# Abrindo os arquivos
# Lista de nomes para facilitar o acesso as pastas e aos arquivos
def plot_8():
	documentos = ['cpf_frente','cpf_tras','rg_frente','rg_tras','cnh_frente','cnh_tras']

	# Laço para repetir o processos anteriores de um vez para todos as amostras boas
	medias = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_bons = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,20):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        doc.append(cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2HSV))
	        docs_bons.append(doc[i])
	        
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_hue = 0
	    sum_sat = 0
	    sum_val = 0
	    
	    for i in range(len(doc)):
	        sum_hue += doc[i][0]
	        sum_sat += doc[i][1]
	        sum_val += doc[i][2]
	        
	    sum_hue = sum_hue/len(doc)
	    sum_sat = sum_sat/len(doc)
	    sum_val = sum_val/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_hue, sum_sat, sum_val]}


	# Laço para repetir o processos anteriores de um vez para todas as amostras ruins
	medias2 = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_ruins = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,7):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        doc.append(cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2HSV))
	        docs_ruins.append(doc[i])
	         
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_hue = 0
	    sum_sat = 0
	    sum_val = 0
	    
	    for i in range(len(doc)):
	        sum_hue += doc[i][0]
	        sum_sat += doc[i][1]
	        sum_val += doc[i][2]
	        
	    sum_hue = sum_hue/len(doc)
	    sum_sat = sum_sat/len(doc)
	    sum_val = sum_val/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias2[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_hue, sum_sat, sum_val]}

	n = len(medias.keys()) + len(medias2.keys()) # Difinindo o número de linhas
	fig , ax = plt.subplots(n, 2, figsize=(10,3*n), gridspec_kw={'width_ratios': [1,3]}) # Criando a "estura" do subplots

	ax.flatten()
	ax = ax.T.flatten()
	for i,doc in zip(range(0,n,2),documentos):
	    ax[0+(i)].imshow(medias[doc]['matrix']); # plotando a imagem boa para ilustrar o documento
	    ax[0+(i)].axis('off'); ax[0+(i)].set_title(doc + ' bom',fontsize=10) 

	    ax[n+(i)].plot(medias[doc]['hist'][0], color='darkorchid') # plotando o histograma
	    ax[n+(i)].axvline(0, linestyle='dashed' ,c='orangered')
	    ax[n+(i)].axvline(100, linestyle='dashed' ,c='orangered')

	    if i == 0 : ax[n+(i)].set_title('Hue',fontsize=10) # coloca o nome no primeiro histograma
	        
	    ax[0+(i+1)].imshow(medias2[doc]['matrix']); # plotando a imagem ruim para ilustrar o documento
	    ax[0+(i+1)].axis('off'); ax[0+(i+1)].set_title(doc + ' ruim',fontsize=10)

	    ax[n+(i+1)].plot(medias2[doc]['hist'][0], color='darkorchid') # plotando o histograma
	    ax[n+(i+1)].axvline(0,linestyle='dashed',c='orangered')
	    ax[n+(i+1)].axvline(100 ,linestyle='dashed',c='orangered')

	plt.show();

#######################################################################################

########################### Notebook 1.10 ############################################

# Lista de nomes para facilitar o acesso as pastas e aos arquivos
def plot_9():
	documentos = ['cpf_frente','cpf_tras','rg_frente','rg_tras','cnh_frente','cnh_tras']

	# Função de plot para facilitar
	def multiplot():
	    n = len(medias.keys()) + len(medias2.keys())
	    fig , ax = plt.subplots(n, 4, figsize=(18,3*n), gridspec_kw={'width_ratios': [1,3,3,3]})

	    cont = 0
	    ax.flatten()
	    ax = ax.T.flatten()
	    for i,doc in zip(range(0,n,2),documentos):
	        ax[0+(i)].imshow(medias[doc]['matrix']); 
	        ax[0+(i)].axis('off'); ax[0+(i)].set_title(doc + ' bom',fontsize=10)

	        ax[n+(i)].plot(medias[doc]['hist'][0], color='red');
	        ax[n*2+(i)].plot(medias[doc]['hist'][1], color='green')
	        ax[n*3+(i)].plot(medias[doc]['hist'][2], color='blue')

	        ax[0+(i+1)].imshow(medias2[doc]['matrix']); 
	        ax[0+(i+1)].axis('off'); ax[0+(i+1)].set_title(doc + ' ruim',fontsize=10)

	        ax[n+(i+1)].plot(medias2[doc]['hist'][0], color='red');
	        ax[n*2+(i+1)].plot(medias2[doc]['hist'][1], color='green')
	        ax[n*3+(i+1)].plot(medias2[doc]['hist'][2], color='blue')

	    plt.show()

	# Laço para repetir o processos anteriores de um vez para todos as amostras boas
	ksize = (70,70) # >> Definindo a máscara do borramento
	medias = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_bons = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,20):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        img_aux = cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2RGB)
	        doc.append(cv2.blur(img_aux, ksize))  # >> Aplicando o borramento
	        docs_bons.append(doc[i])
	        
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_red = 0
	    sum_green = 0
	    sum_blue = 0
	    
	    for i in range(len(doc)):
	        sum_red += doc[i][0]
	        sum_green += doc[i][1]
	        sum_blue += doc[i][2]
	        
	    sum_red = sum_red/len(doc)
	    sum_green = sum_green/len(doc)
	    sum_blue = sum_blue/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_red, sum_green, sum_blue]}

	# Laço para repetir o processos anteriores de um vez para todas as amostras ruins
	medias2 = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_ruins = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,7):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        img_aux = cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2RGB)
	        img_aux = increase_brightness_percent(img_aux, 1.45) # >> Aplicando o brilho percentual
	        img_aux = clahe_func(img_aux) # >> Aplicando a equalização clahe
	        doc.append(cv2.blur(img_aux, ksize))
	        docs_ruins.append(doc[i])
	         
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_red = 0
	    sum_green = 0
	    sum_blue = 0
	    
	    for i in range(len(doc)):
	        sum_red += doc[i][0]
	        sum_green += doc[i][1]
	        sum_blue += doc[i][2]
	        
	    sum_red = sum_red/len(doc)
	    sum_green = sum_green/len(doc)
	    sum_blue = sum_blue/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias2[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_ruins/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_red, sum_green, sum_blue]}

	multiplot()

########################################################################################

################################### Notebook 1.11 ######################################

# Lista de nomes para facilitar o acesso as pastas e aos arquivos
def plot_10():
	documentos = ['cpf_frente','cpf_tras','rg_frente','rg_tras','cnh_frente','cnh_tras']

	# Laço para repetir o processos anteriores de um vez para todos as amostras boas
	medias = {}    # Dicionário para guardar todos as amostras dos documentos e as médias dos histogramas
	docs_bons = [] # Lista para salvar as imagens para utiliza-las mais na frente
	for k in documentos:    # Laço iterando sobre a lista de nomes
	    doc = []            # Lista para salvar os documentos
	    for i in range(0,20):   # Laço para normalizar todos os histogramas do referido documento da vez
	        hist = []     # Lista para salvar os histogramas
	        # Abre e converte o sistema de cores para RGB
	        img = cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(i+1)+').jpg'), cv2.COLOR_BGR2RGB)
	        img = cv2.blur(img, (70,70)) # Borramento
	        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Transformando em HSV
	        doc.append(img)
	        docs_bons.append(doc[i])
	        
	        hist.append(cv2.calcHist(doc[i],[0],None,[256],[0,256]))  # Processo análogo ao anterior para 
	        hist.append(cv2.calcHist(doc[i],[1],None,[256],[0,256]))  # cálcular as média dos histogramas
	        hist.append(cv2.calcHist(doc[i],[2],None,[256],[0,256]))
	        for j in range(0,3):
	            mm = MinMaxScaler()
	            hist[j] = mm.fit_transform(hist[j], hist[j])
	        doc[i] = hist

	    # Média dos histogramas
	    sum_hue = 0
	    sum_sat = 0
	    sum_val = 0
	    
	    for i in range(len(doc)):
	        sum_hue += doc[i][0]
	        sum_sat += doc[i][1]
	        sum_val += doc[i][2]
	        
	    sum_hue = sum_hue/len(doc)
	    sum_sat = sum_sat/len(doc)
	    sum_val = sum_val/len(doc)
	    
	    # Salva o resultado em um dicionário para facilitar nas plotagens
	    medias[k] = {'matrix': cv2.cvtColor(cv2.imread('amostras_boas/'+k+'/'+k+' ('+str(1)+').jpg'), cv2.COLOR_BGR2RGB),
	                 'hist':[sum_hue, sum_sat, sum_val]}

	def media_movel(doc_type, n):
	    medias_suavizada = []
	    tam = len(medias[doc_type]['hist'][0])

	    for i in range(tam-n):
	        soma = 0
	        for j in range(n):
	            soma += medias[doc_type]['hist'][0][i+j][0]
	        medias_suavizada.append(soma/n)
	        
	    for i in range(n,tam-n,-1):
	        soma = 0
	        for j in range(n):
	            soma += medias[doc_type]['hist'][0][i-j][0]
	        medias_suavizada.append(soma/n)
	    
	    return medias_suavizada

	# Plot análogo ao anterior
	n = len(medias.keys())
	k=5

	fig , ax = plt.subplots(n, 3, figsize=(15,3*n), gridspec_kw={'width_ratios': [1,3,3]})

	cont = 0
	ax.flatten()
	ax = ax.T.flatten()
	for i,doc in zip(range(0,n),documentos):
	    ax[0+(i)].imshow(medias[doc]['matrix']); 
	    ax[0+(i)].axis('off'); ax[0+(i)].set_title(doc + ' bom',fontsize=10)
	    
	    ax[n+(i)].plot(medias[doc]['hist'][0], color='mediumblue')
	    if i == 0 : ax[n+(i)].set_title('Hue',fontsize=10)

	    ax[n*2+(i)].plot(media_movel(doc,k), color='darkblue')
	    if i == 0 : ax[n*2+(i)].set_title('Hue (Média Móvel '+str(k)+')',fontsize=10)

	plt.show()

####################################################################################

####################### Notebook Validação dos Modelos #############################

# Link : https://github.com/eduardocassimiro/mandacaru.dev/blob/main/desafio-1/testes-de-hipotese/Valida%C3%A7%C3%A3o%20dos%20Modelos%20de%20Classifica%C3%A7%C3%A3o%20dos%20Histogramas.ipynb

# Função para auxiliar nos plots dos registros de acurácia e loss
def loss_acc_plot(hist, metric, epochs):
    plt.figure(figsize=(12, 7))
    epochs_range = range(1, epochs + 1)
    train_data_vl = hist.history[metric]
    validation_data_vl = hist.history['val_' + metric]
    plt.plot(epochs_range,  train_data_vl, '-o',label='Train Data')
    plt.plot(epochs_range, validation_data_vl, '-o', label='Validation Data')
    plt.legend()
    plt.xticks(range(0, epochs + 1, 5))

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'Training {metric}', fontsize=20)


with open('imagens_originais/desafio1_data.json') as file: 
    data = json.load(file)

X_op2 = deepcopy(data['op2'])
y_3 = deepcopy(data['y3'])

# Criando um dicionário para facilitar a transformação das labels do target em números
labels = {}
for i,j in zip(np.unique(y_3), range(len(np.unique(y_3)))):
    labels[i] = j

# Laço para percorrer todos os valores do target.
# Acessando o valor do elemento da lista do target, ele retorna o nome da sua categoria.
# O nome da categoria é uma chave do dicioário contendo um número para cada categoria.
# O valor do elemento da lista acessa o dicionário que retorna o seu respectivo número e atualiza o valor
# na lista de target.
for i in range(len(y_3)):
    y_3[i] = labels[y_3[i]]

# Codificando a lista do target para o formato que a rede neural do keras aceita
# sem precisarmos configurar algum parâmetro
y_3 = to_categorical(y_3,len(np.unique(y_3)))

# Separando o conjunto de treino e validação
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_op2, y_3, test_size=0.2)

# Separando o conjunto de treino (de fato) e de teste
X__train_1, X_test_1, y__train_1, y_test_1 = train_test_split(X_train_1, y_train_1, test_size=0.2)

model_1 = Sequential() # Definindo o tipo da rede neural
model_1.add(Dense(50, activation='relu', input_shape=(np.array(X_op2)[0].shape))) # Definindo a primeira cada oculta
model_1.add(BatchNormalization()) # Definindo uma camada para normalização em lote para normalizar o treinamento
model_1.add(Dropout(.25)) # Definindo um dropout para retirar os atributos menos relevantes
model_1.add(Dense(3, activation='softmax')) # Definindo a saída
model_1.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['categorical_accuracy'])

def summary_model_1():
	model_1.summary()

history_1 = 0
def fit_model_1():
	global history_1
	history_1 = model_1.fit(np.array(X__train_1), np.array(y__train_1), validation_data=[np.array(X_test_1), np.array(y_test_1)], epochs=20, batch_size=32, verbose=1)

y_pred_1 = 0
def prevision_model_1():
	global y_pred_1
	# Realizando a predição dos dados de validação
	y_pred_1 = model_1.predict(X_val_1) # Predição
	y_pred_1 = [ np.argmax(y) for y in y_pred_1] # O modelo retorna probabilidades, então devemos convertê-las em labels númericas
	y_pred_1 = to_categorical(y_pred_1,len(np.unique(y_pred_1))) # Transformando as labels numéricas nas categorias codificadas
	               # já que para utilizar o classification_report é necessário que o y_pred e Y-val estejam no mesmo formato

def class_report_model_1():
	print(classification_report(y_pred_1, y_val_1, target_names=['CNH', 'CPF', 'RG']))	

def plot_loss_model_1():
	loss_acc_plot(history_1, 'loss', epochs=20)          

def plot_accuracy_model_1():
	loss_acc_plot(history_1,'categorical_accuracy', epochs=20)

# Para utilziar a função ConfusionMatrixDisplay do sklearning, é necessário que as labels estejam
# em um formato númerico o categórico, então precisamos converter os labels novamente


# Plotando a matriz de confusão
def confusion_matrix_model_1():
	y_pred_label_1 = []
	y_val_label_1 = []
	for i in range(len(y_pred_1)):
	    if y_pred_1[i][0] == 1.0:
	        y_pred_label_1.append('CNH')
	    elif y_pred_1[i][1] == 1.0:
	        y_pred_label_1.append('CPF')
	    elif y_pred_1[i][2] == 1.0:
	        y_pred_label_1.append('RG')

	for i in range(len(y_val_1)):
	    if y_val_1[i][0] == 1.0:
	        y_val_label_1.append('CNH')
	    elif y_val_1[i][1] == 1.0:
	        y_val_label_1.append('CPF')
	    elif y_val_1[i][2] == 1.0:
	        y_val_label_1.append('RG')

	cm = confusion_matrix(y_val_label_1, y_pred_label_1, labels=['CNH', 'CPF', 'RG'])
	fig, ax = plt.subplots(figsize=(10, 7))
	ax.set_title('Matriz de Confusão', fontsize=15)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CNH', 'CPF', 'RG']).plot(ax=ax, cmap=plt.cm.Blues)
	plt.show()


##############
X_op2_mod3 = []
for i in range(len(X_op2)):
    X_op2_mod3.append(X_op2[i][-100:])



X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_op2_mod3, y_3, test_size=0.2)

# Separando o conjunto de treino (de fato) e de teste
X__train_2, X_test_2, y__train_2, y_test_2 = train_test_split(X_train_2, y_train_2, test_size=0.2)

X_op2_mod3 = np.array(X_op2_mod3)

model_2 = Sequential()
model_2.add(Dense(20, activation='relu', input_shape=(X_op2_mod3[0].shape)))
model_2.add(BatchNormalization())
model_2.add(Dropout(.25))
model_2.add(Dense(3, activation='softmax'))
model_2.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['categorical_accuracy'])

def summary_model_2():
	model_2.summary()

history_2 = 0
def fit_model_2():
	global history_2
	history_2 = model_2.fit(np.array(X__train_2), np.array(y__train_2), validation_data=[np.array(X_test_2), np.array(y_test_2)], epochs=20, batch_size=32, verbose=1)

y_pred_2 = 0
def prevision_model_2():
	global y_pred_2
	# Realizando a predição dos dados de validação
	y_pred_2 = model_2.predict(X_val_2) # Predição
	y_pred_2 = [ np.argmax(y) for y in y_pred_2] # O modelo retorna probabilidades, então devemos convertê-las em labels númericas
	y_pred_2 = to_categorical(y_pred_2,len(np.unique(y_pred_2))) # Transformando as labels numéricas nas categorias codificadas
	               # já que para utilizar o classification_report é necessário que o y_pred e Y-val estejam no mesmo formato

def class_report_model_2():
	print(classification_report(y_pred_2, y_val_2, target_names=['CNH', 'CPF', 'RG']))	

def plot_loss_model_2():
	loss_acc_plot(history_2, 'loss', epochs=20)          

def plot_accuracy_model_2():
	loss_acc_plot(history_2,'categorical_accuracy', epochs=20)

# Para utilziar a função ConfusionMatrixDisplay do sklearning, é necessário que as labels estejam
# em um formato númerico o categórico, então precisamos converter os labels novamente


# Plotando a matriz de confusão
def confusion_matrix_model_2():
	y_pred_label_2 = []
	y_val_label_2 = []
	for i in range(len(y_pred_2)):
	    if y_pred_2[i][0] == 1.0:
	        y_pred_label_2.append('CNH')
	    elif y_pred_2[i][1] == 1.0:
	        y_pred_label_2.append('CPF')
	    elif y_pred_2[i][2] == 1.0:
	        y_pred_label_2.append('RG')

	for i in range(len(y_val_2)):
	    if y_val_2[i][0] == 1.0:
	        y_val_label_2.append('CNH')
	    elif y_val_2[i][1] == 1.0:
	        y_val_label_2.append('CPF')
	    elif y_val_2[i][2] == 1.0:
	        y_val_label_2.append('RG')

	cm = confusion_matrix(y_val_label_2, y_pred_label_2, labels=['CNH', 'CPF', 'RG'])
	fig, ax = plt.subplots(figsize=(10, 7))
	ax.set_title('Matriz de Confusão', fontsize=15)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CNH', 'CPF', 'RG']).plot(ax=ax, cmap=plt.cm.Blues)
	plt.show()