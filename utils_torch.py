from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import torch


def compare_state_dicts(sd1, sd2):
    # check if the state dicts of two models are the same
    is_equal = True
    for p in sd1:
        if not torch.all(torch.eq(sd1[p], sd2[p])):
            is_equal = False
            break
    return is_equal


def examine_param_space(model, sd1, sd2, train_loader, test_loader, criterion, plot=None):
    """
    Interpolate between the parameters of sd1 and sd2 and evaluate the loss function of the model
    at each point. The resulting plot shows the flatness or sharpness of the minima at each solution.

    See: Goodfellow et al. "Qualitatively Characterizing Neural Network Optimization Problems" (2015)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_train = []
    loss_test = []
    alphas = np.arange(-1, 2.1, 0.1)
    for a in alphas:
        print("alpha: %.1f " % a, end="\r")
        # get model with interpolated weights
        sd_tmp = {p: (1-a)*sd1[p] + a*sd2[p] for p in sd1}
        model.load_state_dict(sd_tmp)
        model = model.to(device)
        # evaluate the model with these parameters on the given data
        model.eval()

        def compute_loss(data_loader):
            tmp_loss = 0
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    # sum up batch loss (criterion should have reduction="sum")
                    tmp_loss += criterion(output, target).item()
            return tmp_loss / len(data_loader.sampler)

        # train
        loss_train.append(compute_loss(train_loader))
        # test
        loss_test.append(compute_loss(test_loader))
    print("alpha: %.1f...done." % a)

    # possibly plot the results
    if plot is not None:
        plt.figure()
        plt.plot(alphas, loss_train, label="$J(\\theta)$ train")
        plt.plot(alphas, loss_test, label="$J(\\theta)$ test")
        plt.xlabel("$\\alpha$")
        plt.ylabel("$J((1-\\alpha)\\cdot\\theta_0 + \\alpha\\cdot\\theta_1)$")
        plt.legend(loc=0)
        plt.title(plot)
    else:
        return alphas, loss_train, loss_test
