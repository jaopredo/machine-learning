import torch

def precision(y: torch.Tensor, t: torch.Tensor) -> float:
    """
    Calcula a precisão do modelo.
    
    Args:
        y (torch.Tensor): Saída do modelo (logits).
        t (torch.Tensor): Rótulos verdadeiros (one-hot).
        
    Returns:
        float: Precisão do modelo.
    """
    # Converte logits em probabilidades usando softmax
    probabilities = torch.softmax(y, dim=1)
    
    # Obtém as classes previstas
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    # Obtém as classes verdadeiras
    true_classes = torch.argmax(t, dim=1)
    
    # Calcula o número de acertos
    correct_predictions = (predicted_classes == true_classes).sum().item()
    
    # Calcula a precisão
    precision_value = correct_predictions / t.size(0)
    
    return precision_value
