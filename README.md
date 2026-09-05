# Machine Learning From Scratch

Este repositório reúne implementações de algoritmos de Machine Learning e Deep Learning desenvolvidas com fins educacionais.

O principal objetivo do projeto é estudar e compreender o funcionamento interno dos modelos, implementando manualmente seus componentes fundamentais em vez de depender exclusivamente das abstrações fornecidas por bibliotecas como PyTorch ou Scikit-Learn.

Embora o projeto utilize PyTorch como infraestrutura para manipulação de tensores e operações matriciais, a lógica dos algoritmos é implementada manualmente para facilitar o aprendizado dos conceitos matemáticos e computacionais envolvidos.

---
# Como instalar

Para o funcionamento do repositório de forma automática, foi criado um arquivo `Makefile` contendo os comandos para instalação. Para preparar seu ambiente
```sh
make setup
```
esse comando automaticamente instala as dependências e detecta se seu computador possui uma placa de vídeo instalada e baixa os pacotes torch adaptados para a sua necessidade (AMD = ROCm, NVIDEA = CUDA, Nenhum = CPU)

Se você possui placa de vídeo, mas não quer utilizar ela para rodar os modelos, você pode forçar o make para fazer o setup de cpu
```sh
make setup-cpu
```

---

# Objetivos

Este repositório foi criado para:

* Estudar algoritmos de Machine Learning e Deep Learning em profundidade.
* Compreender como funciona o treinamento de modelos.
* Implementar forward propagation e backpropagation manualmente.
* Entender o funcionamento de funções de perda e otimizadores.
* Explorar arquiteturas clássicas e modernas.
* Desenvolver uma biblioteca modular inspirada em frameworks reais.
* Servir como material de consulta durante estudos e pesquisas.

---

# Filosofia do Projeto

O objetivo não é competir com bibliotecas de produção.

Em aplicações reais, seria recomendado utilizar:

* PyTorch
* TensorFlow
* JAX
* Scikit-Learn

Entretanto, para aprendizado, muitas abstrações escondem detalhes importantes do funcionamento interno dos modelos.

Por isso, este projeto busca um equilíbrio entre:

* Implementação manual dos algoritmos.
* Organização inspirada em frameworks modernos.
* Código legível e extensível.
* Foco em aprendizado e experimentação.

---

# Estrutura do Projeto

```text
machine-learning/

functions/          # ativações e losses manuais
losses/
optimizers/
trainers/
utils/

notebooks/          # modelos implementados inline em células
│
├── knn.ipynb
├── linear_regression.ipynb
├── logistic_regression.ipynb
├── pca.ipynb
├── mixture_gaussians.ipynb
├── cnn.ipynb
├── gcn.ipynb       # também contém o MultilayerPerceptron
└── ...
```

Os modelos vivem nos notebooks (tensores no `device` definido). Pacotes auxiliares (`utils`, `functions`, `trainers`, `optimizers`) continuam importáveis.

---

# Arquitetura

O projeto segue uma arquitetura modular inspirada em frameworks de Deep Learning.

## Layers

As camadas representam os blocos fundamentais dos modelos.

Exemplos:

* Linear
* ReLU
* Sigmoid
* Tanh
* Softmax
* Graph Convolution Layer

Cada camada é responsável apenas por:

* Forward pass
* Backward pass
* Exposição de parâmetros treináveis

---

## Models

Os modelos são compostos por múltiplas camadas.

Exemplos:

* Logistic Regression
* Multilayer Perceptron (MLP)
* Graph Convolutional Network (GCN)

Os modelos são responsáveis apenas por:

* Definir a arquitetura.
* Encadear as camadas.
* Executar forward e backward propagation.

---

## Loss Functions

As funções de perda são componentes independentes.

Exemplos:

* Mean Squared Error (MSE)
* Binary Cross Entropy (BCE)
* Cross Entropy

Cada loss implementa:

* Cálculo da perda.
* Gradiente da perda.

---

## Optimizers

Os otimizadores realizam a atualização dos parâmetros.

Exemplos:

* SGD
* Adam (quando implementado)

As atualizações são implementadas manualmente para facilitar o entendimento do processo de otimização.

---

# Referências

As implementações deste repositório são baseadas em:

* Artigos científicos originais.
* Livros clássicos de Machine Learning e Deep Learning.
* Documentação oficial do PyTorch.
* Pattern Recognition and Machine Learning (Christopher Bishop).
* Meu outro repositório em conjunto com meus colegas de faculdade sobre as matérias do curso: [Holy Emapian Scripture](https://github.com/Holy-Emapian-Scripture/holy-emapian-scripture)

---

# Aviso

Este projeto possui caráter educacional.

As implementações priorizam aprendizado e compreensão dos algoritmos, podendo diferir das abordagens utilizadas em ambientes de produção.
