% Dynamic Customer Segmentation via Hierarchical Fragmentation-Coagulation
% Processes. This file creates a synthetic dataset and calls hfcp function
% to build HFCP model on this dataset

% define the number of sampling rounds
numberOfSamplingRounds = 100;

% hyperparameters for HFCP
gamma = 0.5;
alpha = 0.8;
epsilon=0.1;

% hyperparameters for Gamma distribution: alpha is shape and beta is scale
gammaAlpha = 2;
gammaBeta = 0.5;

% generate synthetic dataset for demonstration
numberOfTimeSteps = 20;
numberOfProducts = 3;
selectedTargets = 1:numberOfProducts;
numberOfTargets = length(selectedTargets);
numberOfDishes = 3;
numberOfCustomers = 300;

kForCustomers = randi(numberOfDishes, numberOfCustomers, numberOfTimeSteps);
groundTruth = kForCustomers;

% dataMatrix: main purchase data, dimension is the number of customers *
% number of time steps
dataMatrix = random('Poisson',kForCustomers*5,size(kForCustomers));
% sideMatrix: auxiliary information about data matrix, where each row contains the product id and customer id of the corresponding row in data matrix
sideInfoMatrix = [];

% generate synthetic data for sideInfoMatrix
proportionOfCustomersOfProduct = [0.3, 0.5, 0.2];
for i = 1:length(proportionOfCustomersOfProduct)
    numberForI = ceil(numberOfCustomers * proportionOfCustomersOfProduct(i));
    sideInfoMatrix = [sideInfoMatrix; i*ones(numberForI,1),(1:numberForI)'];
end

results = hfcp(dataMatrix,sideInfoMatrix,selectedTargets,numberOfSamplingRounds,gammaAlpha,gammaBeta,gamma,alpha,epsilon);
