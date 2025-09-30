import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import evaluate_model_performance, roc
from model import CNN
from getdata import test_x,test_y
def to_test_model(t5_features, bert_features, esm3_features, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripleFeatureModel().to(device)
    model.load_state_dict(torch.load("./mul_feature_model.pt"))
    model.eval()
    y = torch.tensor(y)

    # Create DataLoader for testing
    test_dataset = TensorDataset(t5_features, bert_features, esm3_features, y)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    pr_list, test_predictions, outs_list = [], [], []
    with torch.no_grad():
        for batch_t5, batch_bert, batch_esm3, batch_labels in test_dataloader:
            batch_t5 = batch_t5.to(device)
            batch_bert = batch_bert.to(device)
            batch_esm3 = batch_esm3.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_t5, batch_bert, batch_esm3)
            outputs = torch.nn.functional.softmax(outputs, -1)
            outs_list.extend(outputs)
            test_predictions.extend(torch.argmax(outputs, dim=-1).tolist())
            pr_list.extend(outputs[:, 1].cpu().numpy())

    evaluate_model_performance(y, test_predictions)
    roc(y, pr_list)
    mcc = matthews_corrcoef(y, test_predictions)
    acc = accuracy_score(y, test_predictions)
    return outs_list, mcc, acc

if __name__ == "__main__":
    outs, _, _ = to_test_model(test_t5, test_Bert, test_ESM3, test_y)