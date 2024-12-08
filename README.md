🌟💡 **深度学习与神经网络**

## 🎯 项目概述

本项目包含两个主要部分：**深度学习销售预测**📈和**神经网络代理**🤖。通过数据科学、机器学习与神经网络技术，展示了如何应用深度学习模型来预测商店销售数据，并使用神经网络代理来模拟游戏决策。最终的目标是优化商店运营和游戏行为预测。以下是每个模块的详细介绍。

---

## 游戏截图
![1.png](img/1.png)
![2.png](img/2.png)

## 📊💻 深度学习销售预测

### 🛒 项目背景  
本模块基于德国某商店的历史销售数据，利用深度学习模型预测未来12个月的销售营业额。通过对多个模型的横向比较，最终选定 **CatBoost** 作为最佳优化模型，显著提高了预测精度。

### 🔧 主要技术  
- **数据集**：使用德国商店的历史销售数据。  
- **模型**：采用多种深度学习模型（线性回归、神经网络等）进行对比，最终选用 **CatBoost**。  
- **优化效果**：通过模型优化，将 **RMSE** 从 1800 优化至 700，实现了精准的销售额预测。

### ⚙️ 使用技术  
- **Python版本**：3.12.7  
- **深度学习框架**： **CatBoost**  
- **数据处理**： **pandas**, **numpy**

### 🔄 实现步骤  
1. **数据预处理**：数据清洗、缺失值处理和特征工程。  
2. **模型训练与优化**：对比多种模型，最终选定 **CatBoost** 模型并优化。  
3. **性能评估**：通过 **RMSE**（均方根误差）评估模型准确性。  
4. **模型应用**：最终模型预测商店未来12个月的销售额，帮助商店做出更合理的运营决策。

---

## 🤖🎮 神经网络代理

### 🎮 项目背景  
本模块使用神经网络代理来模拟游戏中的决策行为。代理通过**Data Collection**模式收集原始游戏数据，划分训练集和测试集，学习如何基于当前状态做出正确的决策，最终实现目标位置的预测和操作。

### 🔧 主要技术  
- **数据采集**：通过 **Data Collection** 模式收集游戏中的原始数据。  
- **神经网络结构**：三层神经网络，输入层有4个神经元，隐藏层有5个神经元，输出层为2个神经元。  
- **训练方法**：使用前向传播计算损失函数（均方误差），并采用学习率0.01的带动量梯度下降算法进行反向传播训练。  
- **训练数据**：最初使用2000条数据，随着数据量增加至10万条，网络预测精度显著提升。

### ⚙️ 使用技术  
- **Python版本**：3.12.7  
- **神经网络框架**： **自定义神经网络实现**  
- **数据处理**： **pandas**, **numpy**

### 🧠 网络结构  
- **输入层**：4个神经元，代表游戏状态特征（如位置、速度等）。  
- **隐藏层**：5个神经元，激活函数使用 **Sigmoid**。  
- **输出层**：2个神经元，表示目标位置的预测值（如X、Y坐标）。  
- **激活函数**：隐藏层使用 **Sigmoid**，输出层使用线性激活函数。

### 📚 训练过程  
1. **前向传播**：计算网络输出，得到预测值。  
2. **反向传播**：计算损失函数（均方误差），通过梯度下降更新权重和偏置。  
3. **学习率与动量**：使用学习率0.01，结合动量进行加速收敛。  
4. **数据集划分**：初期使用2000条数据，数据量增加到10万条后，网络预测能力显著提升。

### 🚀 结果与展望  
尽管在初期数据较少时，神经网络表现不佳，但随着数据量的增加，网络逐渐能够准确预测游戏目标的落点，并作出相应的操作。未来将继续优化网络结构，增加数据量，探索更复杂的算法以提高模型准确度。

---

## 📝 代码实现

### 📊 **deep** 文件夹  
- **任务**：通过多种深度学习模型（包括CatBoost）进行销售额预测。  
- **实现细节**：比较多个深度学习模型，选用 **CatBoost** 模型，并对其进行优化，最终取得了良好的预测效果。

### 🎮 **neural_network** 文件夹  
- **任务**：构建一个神经网络代理，模拟并预测游戏中的目标位置。  
- **实现细节**：设计了三层神经网络（4维输入层，5维隐藏层，2维输出层），通过前向传播计算损失，使用带动量的梯度下降法进行训练。

### 📚 使用的Python库  
- **NumPy**：用于数值计算与矩阵操作。  
- **pandas**：用于数据处理与清洗。  
- **CatBoost**：用于训练和优化深度学习模型。  
- **joblib**：用于保存和加载模型权重。

---

## 🖥️ 运行环境

- **Python版本**：3.12.7  
- **依赖库**：确保安装以下依赖：  
  - numpy  
  - pandas  
  - catboost  
  - joblib  

---

## 🚀 总结与展望

本项目结合了深度学习与神经网络技术，展示了如何使用机器学习模型进行销售预测以及如何通过神经网络代理模拟游戏中的决策行为。销售预测模块通过 **CatBoost** 模型取得了优秀的预测效果，神经网络代理则在数据量增加后实现了较为精准的目标预测。未来，我们将通过优化神经网络模型、增加训练数据和探索更多算法来提升项目的整体性能和应用潜力。

---
QQ:284190056(付费咨询200起)