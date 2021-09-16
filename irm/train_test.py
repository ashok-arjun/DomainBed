import torch
import wandb
import torch.nn.functional as F


def test_model(model, device, test_loader, set_name="test_set", epoch=-1):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('Performance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  wandb.log({str(set_name) +"/loss": test_loss}, step=epoch)
  wandb.log({str(set_name) +"/acc": 100. * correct / len(test_loader.dataset)}, step=epoch)

  return 100. * correct / len(test_loader.dataset)
