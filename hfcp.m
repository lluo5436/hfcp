function results = hfcp(dataMatrix,sideInfoMatrix,selectedTargets,numberOfSamplingRounds,gammaAlpha,gammaBeta,gamma,alpha,epsilon)

    numberOfActiveCustomers = size(dataMatrix,1);
    numberOfDays = size(dataMatrix,2);
    numberOfTargets = length(selectedTargets);

    % use to initialise non-parametric variables such as beta and lambda
    numberOfMaxDishes = floor(numberOfActiveCustomers/10);
    numberOfMaxGroups = floor(numberOfActiveCustomers/5);
    numberOfGroupsForAllRounds = zeros(numberOfSamplingRounds,numberOfDays);

    % Part 1) declare variables and initialisation
    % should make sure that the rhoCurrent has more clusters than piCurrent and
    % groups in rhoCurrent must be fragments of groups in piCurrent.
    piCurrent = ones(numberOfActiveCustomers,numberOfDays);
    rhoCurrent = ones(numberOfActiveCustomers,numberOfDays-1);
    % roughly estimated number of "dishes" and "tables" based on prior knowledge
    numberOfTablesForEachTarget = zeros(numberOfTargets,1);
    for i = 1:numberOfTargets
        target = selectedTargets(i);
        numberOfTablesForEachTarget(i,1) = alpha * log(1+length(find(sideInfoMatrix(:,1)==target))/alpha);
    end
    numberOfTables = sum(numberOfTablesForEachTarget);
    numberOfDishes = gamma * log(1+numberOfTables/gamma);

    % initialisation: assign customers to CRP blocks (i.e. select tables in HDP)
    % ONLY in initialisation: assume each k corresponds to one block
    numberOfDishes = ceil(numberOfDishes);
    ztemp = randi(numberOfDishes,numberOfActiveCustomers,1);
    piCurrent = piCurrent .* ztemp;
    rhoCurrent = rhoCurrent .* ztemp;

    % keep a record of customer-dish assignment, which can facilitate
    % dish-level analysis
    kForCustomers = piCurrent;

    beta = -1 * ones(numberOfDays, numberOfMaxDishes);
    beta(:,1:(numberOfDishes+1)) = drchrnd([ones(1, numberOfDishes) * (numberOfTables/numberOfDishes), gamma],numberOfDays);

    % count number of valid dishes based on beta, including one new dish (corresponds to beta_u)
    numberOfDishesT = zeros(numberOfDays,1);
    for currentT = 1:numberOfDays
        numberOfDishesT(currentT,1) = length(find(beta(currentT,:)~=-1));
    end

    logLikelihood = zeros(numberOfSamplingRounds,1);

    % Part 2) Gibbs sampling to infer parameters and partitions for all time steps
    for round = 1:numberOfSamplingRounds
        disp(['round:', num2str(round)]);

        % final lambda for this round, which can be used to compute log
        % likelihood
        lambdaDishTAll = -1 * ones(numberOfDays,numberOfMaxDishes);
        for currentT = 1:numberOfDays
            allDishesT = unique(kForCustomers(:,currentT));
            nAllDishesT = length(allDishesT);

            for index = 1:nAllDishesT
                dish = allDishesT(index,1);
                numerator = sum(dataMatrix(kForCustomers(:,currentT) == dish,currentT));
                denominator = length(dataMatrix(kForCustomers(:,currentT) == dish,currentT));
                lambdaDishTAll(currentT,index) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
            end
        end

        % compute loglikelihood for all data in this iteration
        currentLogLikelihood = 0;
        for currentT = 1:numberOfDays
            allDishesT = unique(kForCustomers(:,currentT));
            for customer = 1:numberOfActiveCustomers
                dish = find(allDishesT ==kForCustomers(customer,currentT));
                tempXT = dataMatrix(customer,currentT);
                delta = exp(-lambdaDishTAll(currentT,dish))*(lambdaDishTAll(currentT,dish)^tempXT)/factorial(tempXT);
                currentLogLikelihood = currentLogLikelihood + log(delta);

            end
        end
        logLikelihood(round,1) = currentLogLikelihood;

        % Step 1: sampling table and selecting dish (c,z)
        % go through each product
        for target = selectedTargets
            targetRange = find(sideInfoMatrix(:,1)==target);
            numberOfTargetCustomers = length(targetRange);

            disp(['product: ', num2str(target)]);

            % go through each customer
            for customer = 1:numberOfTargetCustomers
                customerGlobalIndex = targetRange(customer);
                excludeCurrentCustomerGlobal = 1:numberOfActiveCustomers;
                excludeCurrentCustomerGlobal(customerGlobalIndex) = [];

                x_ji = dataMatrix(customerGlobalIndex,:);

                % disp(['product: ', num2str(target), ' customer: ', num2str(customer), ' global index: ', num2str(customerGlobalIndex)]);
                mc = zeros(numberOfDays,numberOfTargetCustomers);
                mf = zeros(numberOfDays-1,numberOfTargetCustomers);
                % exclude current customer and also in the target range
                excludeCurrentCustomer = targetRange;
                excludeCurrentCustomer(customer) = [];

                % update lambdaTA for every customer
                % for simplicity, lambda is computed for each block, but based on all blocks with the same dish
                % notice that there are blocks with the same lambda_k
                % preparing lambdaTA in this way allows to use "block" index in the following computation
                lambdaTA = -1 * ones(numberOfDays, numberOfMaxGroups);
                for currentT = 1:numberOfDays
                    allGroupsPiTWithEmpty = unique([-1; piCurrent(excludeCurrentCustomer,currentT)]);
                    nAllGroupsPiTWithEmpty = length(allGroupsPiTWithEmpty);

                    % start from the non-empty block
                    if nAllGroupsPiTWithEmpty > 1
                        for index = 2:nAllGroupsPiTWithEmpty
                            group = allGroupsPiTWithEmpty(index,1);
                            temp = kForCustomers(excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT) == group),currentT);

                            dish = temp(1,1);
                            % find all rows exluding the current customer with the same dish
                            relevantRows = excludeCurrentCustomerGlobal(kForCustomers(excludeCurrentCustomerGlobal,currentT) == dish);
                            % MAP for gamma-poisson
                            numerator = sum(dataMatrix(relevantRows,currentT));
                            denominator = length(relevantRows);
                            lambdaTA(currentT,index) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
                        end
                    end
                end

                % lambda for selecting a new table, the dish can be the past K
                % dishes or a new dish
                % K + 1 cases, the new dish is at the end (to be consistent with beta)
                lambdaforEmptyTable = -1 * ones(numberOfDays, numberOfMaxDishes);
                for currentT = 1:numberOfDays
                    % if use unique here, have to keep updating lambda and
                    % beta, whenever a dish or a block disappears
                    allDishesT = unique(kForCustomers(:,currentT));

                    for dishIndex = 1:length(allDishesT)
                        dish = allDishesT(dishIndex,1);
                        relevantRows = excludeCurrentCustomerGlobal(kForCustomers(excludeCurrentCustomerGlobal,currentT) == dish);
                        % MAP for gamma-poisson
                        numerator = sum(dataMatrix(relevantRows,currentT));
                        denominator = length(relevantRows);
                        lambdaforEmptyTable(currentT,dishIndex) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
                    end

                    % new dish: all customers except the current
                    numerator = sum(dataMatrix(:,currentT))-x_ji(currentT);
                    denominator = numberOfActiveCustomers-1;
                    lambdaforEmptyTable(currentT,length(allDishesT)+1) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
                end

                % Part 2.1) backwards filtering
                % compute mc and mf for each time step based on the current
                % partition
                allGroupsPiTWithEmpty = unique([-1; piCurrent(excludeCurrentCustomer,numberOfDays)]);
                for currentT = numberOfDays:-1:1

                    % base case
                    if currentT == numberOfDays
                        mc(numberOfDays,:) = 1;
                        % recursive cases
                    else
                        tempXT1 = x_ji(currentT+1);
                        % go through each partition a \in piCurrent(all rows excluding current customer,currentT+1),
                        % and use -1 to lable the empty set, which MAY or MAY NOT
                        % exist in the original set of clusters in piCurrent!!
                        allGroupsPiT1WithEmpty = allGroupsPiTWithEmpty;
                        allGroupsPiTWithEmpty = unique([-1; piCurrent(excludeCurrentCustomer,currentT)]);
                        allGroupsRhoTWithEmpty = unique([-1; rhoCurrent(excludeCurrentCustomer,currentT)]);
                        nAllGroupsPiT1WithEmpty = length(allGroupsPiT1WithEmpty);
                        nAllGroupsPiTWithEmpty = length(allGroupsPiTWithEmpty);
                        nAllGroupsRhoTWithEmpty = length(allGroupsRhoTWithEmpty);

                        nAllDishesT1WithEmpty = numberOfDishesT(currentT+1,1);

                        % update mf(currentT,indexBT)
                        for indexBT = 1:nAllGroupsRhoTWithEmpty

                            % b - current b in this iteration, update mf(currentT,indexBT)
                            b = allGroupsRhoTWithEmpty(indexBT,1);
                            bSingleton = (b==-1);
                            delta = 0;

                            % compute mf at currentT based on Equation (14)
                            for indexAT1 = 1:nAllGroupsPiT1WithEmpty
                                % a1 - possible group a at currentT+1
                                a1 = allGroupsPiT1WithEmpty(indexAT1,1);
                                a1Singleton = (a1==-1);

                                % likelihood
                                if a1Singleton
                                    % consider as a weighted likelihood of K+1
                                    % different cases
                                    for emptyCase = 1:nAllDishesT1WithEmpty
                                        tempLambda = lambdaforEmptyTable(currentT+1,emptyCase);
                                        delta = delta + beta(currentT+1, emptyCase) * (exp(-tempLambda)*(tempLambda^tempXT1)/factorial(tempXT1));
                                    end
                                else
                                    delta = exp(-lambdaTA(currentT+1,indexAT1))*(lambdaTA(currentT+1,indexAT1)^tempXT1)/factorial(tempXT1);
                                end

                                % transition probablity Equation (13)
                                % groups in allGroupsRhoT that coagulate to a1
                                if ~a1Singleton
                                    customersInA1 = excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT+1)==a1);
                                    coagToA = unique(rhoCurrent(customersInA1,currentT));
                                    nCoagToA = length(find(coagToA>0));

                                end

                                if a1Singleton && bSingleton
                                    % since b is singleton, so #rho_t^-i in the
                                    % paper should be nAllGroupsRhoTWithEmpty-1.
                                    pab = alpha/(alpha+epsilon*(nAllGroupsRhoTWithEmpty-1));
                                elseif ~a1Singleton && bSingleton
                                    pab = epsilon*nCoagToA/(alpha+epsilon*(nAllGroupsRhoTWithEmpty-1));
                                elseif ~a1Singleton && ~bSingleton && ismember(b,coagToA)
                                    pab = 1;
                                else
                                    pab = 0;
                                end

                                mf(currentT,indexBT) = mf(currentT,indexBT) + mc(currentT+1,indexAT1) * delta * pab;
                            end
                        end

                        % compute mc at time currentT based on Equation (15),
                        % update mc(currentT, all indexAT)
                        for indexAT = 1:nAllGroupsPiTWithEmpty
                            % a - one possible group at currentT
                            % aSingleton - true or false, whether a is a singleton
                            % cluster
                            a = allGroupsPiTWithEmpty(indexAT,1);
                            aSingleton = (a==-1);

                            % using the mf(currentT, indexBT) which has been
                            % computed
                            for indexBT = 1:nAllGroupsRhoTWithEmpty
                                % b - current b in this iteration, update mf(currentT,indexBT)
                                b = allGroupsRhoTWithEmpty(indexBT,1);
                                bSingleton = (b==-1);

                                % transition probablity Equation (12)
                                % number of elements in a and b, excluding current
                                % customer i
                                nA = length(find(piCurrent(excludeCurrentCustomer,currentT)==a));
                                nB = length(find(rhoCurrent(excludeCurrentCustomer,currentT)==b));
                                % groups in possibleGroupsRhoT that formed by
                                % fragments of a
                                if ~aSingleton
                                    customersInA = excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT)==a);
                                    fragFromA = unique(rhoCurrent(customersInA,currentT));
                                    nFragFromA = length(find(fragFromA>0));

                                end

                                if aSingleton && bSingleton
                                    pba = 1;
                                elseif ~aSingleton && bSingleton
                                    pba = epsilon*nFragFromA/nA;
                                elseif ~aSingleton && ~bSingleton && ismember(b,fragFromA)
                                    pba = (nB-epsilon)/nA;
                                else
                                    pba = 0;
                                end
                                mc(currentT,indexAT) = mc(currentT,indexAT) + mf(currentT,indexBT) * pba;

                            end
                        end
                    end
                end

                % Part 2.2) forwards sampling
                for currentT = 1:numberOfDays-1

                    tempXT = x_ji(currentT);
                    tempXT1 = x_ji(currentT+1);

                    % compute a for t = 1
                    if currentT == 1
                        allGroupsPiTWithEmpty = unique([-1; piCurrent(excludeCurrentCustomer,currentT)]);
                        nAllGroupsPiTWithEmpty = length(allGroupsPiTWithEmpty);

                        nAllDishesTWithEmpty = numberOfDishesT(currentT,1);

                        % compute probability of each case
                        % the first empty case store the new dish
                        % from nAllGroupsPiTWithEmpty + 1 to the end, store all cases for K previous dishes
                        paForSampling = zeros(nAllGroupsPiTWithEmpty+nAllDishesTWithEmpty-1,1);
                        for indexAT = 1:nAllGroupsPiTWithEmpty
                            a = allGroupsPiTWithEmpty(indexAT,1);

                            nA = length(find(piCurrent(excludeCurrentCustomer,currentT)==a));
                            aSingleton = (a==-1);

                            if aSingleton
                                crpTemp = alpha /(numberOfTargetCustomers-1+alpha);
                            else
                                crpTemp = nA/(numberOfTargetCustomers-1+alpha);
                            end

                            % likelihood
                            if aSingleton
                                % only deal with new dish, multiply beta_u here
                                tempLambda = lambdaforEmptyTable(currentT, nAllDishesTWithEmpty);
                                delta = beta(currentT,nAllDishesTWithEmpty)* exp(-tempLambda)*(tempLambda^tempXT)/factorial(tempXT);
                            else
                                delta = exp(-lambdaTA(currentT,indexAT))*(lambdaTA(currentT,indexAT)^tempXT)/factorial(tempXT);
                            end
                            paForSampling(indexAT,1) = crpTemp * delta * mc(currentT,indexAT);
                        end

                        % deal with the other K cases (previous dishes) for a new table
                        emptyCasePaForSampling = zeros(nAllDishesTWithEmpty-1,1);
                        crpTemp = alpha /(numberOfTargetCustomers-1+alpha);
                        for emptyCase = 1:(nAllDishesTWithEmpty-1)
                            tempLambda = lambdaforEmptyTable(currentT,emptyCase);
                            delta = beta(currentT, emptyCase) * (exp(-tempLambda)*(tempLambda^tempXT)/factorial(tempXT));
                            emptyCasePaForSampling(emptyCase,1) = crpTemp * delta * mc(currentT,indexAT);
                        end
                        paForSampling((nAllGroupsPiTWithEmpty+1):end) = emptyCasePaForSampling;

                        % sample based on paForSampling using sample function
                        if ~any(paForSampling)
                            error('error paForSampling');
                        else
                            temp = sample(paForSampling,1);
                        end

                        % for table:
                        % if the sample result is a new table (first or last K cases) - singleton, it should create
                        % a new label
                        if temp == 1 || temp > nAllGroupsPiTWithEmpty
                            newA = max(piCurrent(targetRange,currentT))+1;
                        else
                            newA = allGroupsPiTWithEmpty(temp,1);
                        end

                        % for dish:
                        allDishesT = unique(kForCustomers(:,currentT));
                        if temp == 1
                            newDish = max(kForCustomers(:,currentT)) + 1;

                            % update beta and numberOfDishesT
                            % insert tempB * betaU, and the new betaU = (1-tempB) * betaU
                            tempB = betarnd(1,gamma);
                            tempBeta = beta(currentT, 1:nAllDishesTWithEmpty);
                            betaU = tempBeta(end);
                            tempBeta = [tempBeta(1:(end-1)), tempB * betaU, (1-tempB) * betaU];
                            beta(currentT,1:(nAllDishesTWithEmpty+1)) = tempBeta;

                            numberOfDishesT(currentT,1) = numberOfDishesT(currentT,1) + 1;
                        elseif temp > nAllGroupsPiTWithEmpty
                            newDish = allDishesT((temp - nAllGroupsPiTWithEmpty),1);
                        else
                            group = allGroupsPiTWithEmpty(temp,1);
                            customersInTempBlock = kForCustomers(excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT) == group),currentT);

                            newDish = customersInTempBlock(1,1);
                        end

                        % update the target (customer, currentT)
                        piCurrent(targetRange(customer),currentT) = newA;

                        oldDish = kForCustomers(targetRange(customer),currentT);
                        kForCustomers(targetRange(customer),currentT) = newDish;

                        % if customer is the only customer with its old dish,
                        % remove that dish
                        oldDishDisappears = isempty(find(kForCustomers(:,currentT)==oldDish, 1));
                        if oldDishDisappears
                            kToRemove = find(allDishesT(:)==oldDish);
                            betaOldRow = beta(currentT,:);
                            betaOldRow(kToRemove) = [];
                            betaNew = [betaOldRow,-1];
                            beta(currentT,:) = betaNew;
                            numberOfDishesT(currentT,1) = numberOfDishesT(currentT,1) - 1;
                        end
                    end

                    % for the following time steps t>1
                    % the following computation should be done for all cases
                    % 1:(numberOfDays-1)
                    allGroupsPiT1WithEmpty = unique([-1; piCurrent(excludeCurrentCustomer,currentT+1)]);
                    allGroupsRhoTWithEmpty = unique([-1; rhoCurrent(excludeCurrentCustomer,currentT)]);
                    nAllGroupsPiT1WithEmpty = length(allGroupsPiT1WithEmpty);
                    nAllGroupsRhoTWithEmpty = length(allGroupsRhoTWithEmpty);

                    nAllDishesT1WithEmpty = numberOfDishesT(currentT+1,1);

                    % sample b for currentT
                    a = piCurrent(targetRange(customer),currentT);
                    % different from filtering
                    aSingleton = (length(find(piCurrent(targetRange,currentT)==a))==1);

                    pbaForSampling = zeros(nAllGroupsRhoTWithEmpty,1);
                    for indexBT = 1:nAllGroupsRhoTWithEmpty
                        b = allGroupsRhoTWithEmpty(indexBT,1);
                        bSingleton = (b==-1);

                        % transition probablity Equation (12)
                        % groups in possibleGroupsRhoT that formed by
                        % fragments of a
                        % both customersInA and customersInCheck exclude current
                        % customer
                        if ~aSingleton
                            customersInA = excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT)==a);
                            fragFromA = unique(rhoCurrent(customersInA,currentT));
                            nFragFromA = length(find(fragFromA>0));

                        end
                        % number of elements in a and b, excluding current
                        % customer i
                        nA = length(find(piCurrent(excludeCurrentCustomer,currentT)==a));
                        nB = length(find(rhoCurrent(excludeCurrentCustomer,currentT)==b));

                        if aSingleton && bSingleton
                            pbaTemp = 1;
                        elseif ~aSingleton && bSingleton
                            pbaTemp = epsilon*nFragFromA/nA;
                        elseif ~aSingleton && ~bSingleton && ismember(b,fragFromA)
                            pbaTemp = (nB-epsilon)/nA;
                        else
                            pbaTemp = 0;
                        end
                        pbaForSampling(indexBT,1) = pbaTemp * mf(currentT,indexBT);
                    end

                    % sample based on pbaForSampling
                    if ~any(pbaForSampling)
                        error('error pbaForSampling');
                    else
                        temp = sample(pbaForSampling,1);
                    end
                    % if the sample result is the first case - singleton, it should create
                    % a new label
                    if temp == 1
                        newB = max(rhoCurrent(targetRange,currentT))+1;
                    else
                        newB = allGroupsRhoTWithEmpty(temp,1);
                    end

                    rhoCurrent(targetRange(customer),currentT) = newB;

                    % sample a for currentT+1 based on new b
                    b = newB;
                    bSingleton = (length(find(rhoCurrent(targetRange,currentT)==b))==1);

                    pabForSampling = zeros(nAllGroupsPiT1WithEmpty,1);
                    for indexAT1 = 1:nAllGroupsPiT1WithEmpty
                        a1 = allGroupsPiT1WithEmpty(indexAT1,1);
                        a1Singleton = (a1==-1);

                        % transition probablity Equation (13)
                        % groups in possibleGroupsRhoT that coagulate to a
                        if ~a1Singleton
                            customersInA1 = excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT+1)==a1);
                            coagToA = unique(rhoCurrent(customersInA1,currentT));
                            nCoagToA = length(find(coagToA>0));

                        end

                        if a1Singleton && bSingleton
                            pabTemp = alpha/(alpha+epsilon*(nAllGroupsRhoTWithEmpty-1));
                        elseif ~a1Singleton && bSingleton
                            pabTemp = epsilon*nCoagToA/(alpha+epsilon*(nAllGroupsRhoTWithEmpty-1));
                        elseif ~a1Singleton && ~bSingleton && ismember(b,coagToA)
                            pabTemp = 1;
                        else
                            pabTemp = 0;
                        end

                        % likelihood
                        if a1Singleton
                            delta = 0;
                            if bSingleton
                                % consider as a weighted likelihood of K+1
                                % different cases
                                for emptyCase = 1:nAllDishesT1WithEmpty
                                    tempLambda = lambdaforEmptyTable(currentT+1,emptyCase);
                                    delta = delta + beta(currentT+1, emptyCase) * (exp(-tempLambda)*(tempLambda^tempXT1)/factorial(tempXT1));
                                end
                            else
                                if pabTemp~=0
                                    disp('error');
                                end
                            end
                        else
                            delta = exp(-lambdaTA(currentT+1,indexAT1))*(lambdaTA(currentT+1,indexAT1)^tempXT1)/factorial(tempXT1);
                        end

                        pabForSampling(indexAT1,1) = pabTemp * delta * mc(currentT+1,indexAT1);
                    end

                    % sample based on pabForSampling
                    if ~any(pabForSampling)
                        error('error pabForSampling');
                    else
                        temp = sample(pabForSampling,1);
                    end
                    % if the sample result is the first case - singleton, it should create
                    % a new label
                    if temp == 1
                        newA1 = max(piCurrent(targetRange,currentT+1))+1;
                    else
                        newA1 = allGroupsPiT1WithEmpty(temp,1);
                    end
                    % for dish:
                    allDishesT1 = unique(kForCustomers(:,currentT+1));

                    if temp == 1
                        % sample based on emptyCasePabForSampling
                        % deal with K+1 cases: K previous dishes or a new dish for a new table
                        emptyCasePabForSampling = zeros(nAllDishesT1WithEmpty,1);
                        pabTemp = alpha/(alpha+epsilon*(nAllGroupsRhoTWithEmpty-1));
                        for emptyCase = 1:nAllDishesT1WithEmpty
                            tempLambda = lambdaforEmptyTable(currentT+1,emptyCase);
                            delta = beta(currentT+1, emptyCase) * (exp(-tempLambda)*(tempLambda^tempXT1)/factorial(tempXT1));
                            emptyCasePabForSampling(emptyCase,1) = pabTemp * delta * mc(currentT+1,indexAT1);
                        end

                        dishSampleTemp = sample(emptyCasePabForSampling);

                        if dishSampleTemp == nAllDishesT1WithEmpty
                            newDish = max(kForCustomers(:,currentT+1)) + 1;

                            % update beta and numberOfDishesT
                            % insert tempB * betaU, and the new betaU = (1-tempB) * betaU
                            tempB = betarnd(1,gamma);
                            tempBeta = beta(currentT+1, 1:nAllDishesT1WithEmpty);
                            betaU = tempBeta(end);
                            tempBeta = [tempBeta(1:(end-1)), tempB * betaU, (1-tempB) * betaU];
                            beta(currentT+1,1:(nAllDishesT1WithEmpty+1)) = tempBeta;

                            numberOfDishesT(currentT+1,1) = numberOfDishesT(currentT+1,1) + 1;
                        else
                            newDish = allDishesT1(dishSampleTemp,1);
                        end
                    else
                        group = allGroupsPiT1WithEmpty(temp,1);
                        customersInTempBlock = kForCustomers(excludeCurrentCustomer(piCurrent(excludeCurrentCustomer,currentT+1) == group),currentT+1);

                        newDish = customersInTempBlock(1,1);
                    end

                    piCurrent(targetRange(customer),currentT+1) = newA1;

                    oldDish = kForCustomers(targetRange(customer),currentT+1);
                    kForCustomers(targetRange(customer),currentT+1) = newDish;

                    % if customer is the only customer with its old dish,
                    % remove that dish
                    oldDishDisappears = isempty(find(kForCustomers(:,currentT+1)==oldDish, 1));
                    if oldDishDisappears
                        kToRemove = find(allDishesT1(:)==oldDish);
                        betaOldRow = beta(currentT+1,:);
                        betaOldRow(kToRemove) = [];
                        betaNew = [betaOldRow,-1];
                        beta(currentT+1,:) = betaNew;
                        numberOfDishesT(currentT+1,1) = numberOfDishesT(currentT+1,1) - 1;
                    end
                end
            end
        end

        % final lambda for this round, which can be used to compute log
        % likelihood
        lambdaDishTAll = -1 * ones(numberOfDays,numberOfMaxDishes);
        for currentT = 1:numberOfDays
            allDishesT = unique(kForCustomers(:,currentT));
            nAllDishesT = length(allDishesT);

            for index = 1:nAllDishesT
                dish = allDishesT(index,1);
                numerator = sum(dataMatrix(kForCustomers(:,currentT) == dish,currentT));
                denominator = length(dataMatrix(kForCustomers(:,currentT) == dish,currentT));
                lambdaDishTAll(currentT,index) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
            end
        end

        % compute loglikelihood for all data in this iteration
        currentLogLikelihood = 0;
        for currentT = 1:numberOfDays
            allDishesT = unique(kForCustomers(:,currentT));
            for customer = 1:numberOfActiveCustomers
                dish = find(allDishesT ==kForCustomers(customer,currentT));
                tempXT = dataMatrix(customer,currentT);
                delta = exp(-lambdaDishTAll(currentT,dish))*(lambdaDishTAll(currentT,dish)^tempXT)/factorial(tempXT);
                currentLogLikelihood = currentLogLikelihood + log(delta);

            end
        end

        % Step 2: sampling/updating dish k for each table
        % sample dish for a table, can either be an existing or a new one
        disp('sample dish for each table');
        for currentT = 1:numberOfDays
            % compute probability for each product
            for target = selectedTargets
                targetRange = find(sideInfoMatrix(:,1)==target);

                % sample for each table of this product
                allGroupsPiT = unique(piCurrent(targetRange,currentT));
                nAllGroupsPiT = length(allGroupsPiT);

                for tableIndex = 1:nAllGroupsPiT
                    table = allGroupsPiT(tableIndex);
                    relevantRows = targetRange(piCurrent(targetRange,currentT)==table);

                    excludeCurrentTableGlobal = 1:numberOfActiveCustomers;
                    excludeCurrentTableGlobal = find(~ismember(excludeCurrentTableGlobal,relevantRows));

                    allDishesT = unique(kForCustomers(:,currentT));
                    nDishesT = length(allDishesT);

                    xCurrentTable = dataMatrix(relevantRows,currentT);
                    pSelectDishes = zeros(nDishesT+1,1);

                    lambdaDishT = zeros(1,nDishesT);
                    for dishK = 1:nDishesT
                        rowsDishKNotCurrentTable = excludeCurrentTableGlobal(kForCustomers(excludeCurrentTableGlobal,currentT)==allDishesT(dishK));
                        numerator = sum(dataMatrix(rowsDishKNotCurrentTable,currentT));
                        denominator = length(rowsDishKNotCurrentTable);
                        lambdaDishT(1,dishK) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
                    end

                    % for existing dishes, use beta instead of counting tables
                    for dishK = 1:nDishesT
                        delta = 1;
                        for tempX = xCurrentTable'
                            delta = delta * exp(-lambdaDishT(1,dishK))*(lambdaDishT(1,dishK)^tempX)/factorial(tempX);
                        end
                        if delta < realmin
                            delta = realmin * 10^10;
                        end
                        pSelectDishes(dishK,1) = beta(currentT,dishK) * delta;
                    end

                    % for a new dish
                    delta = 1;
                    % compute lambda without the current table
                    numerator = sum(dataMatrix(excludeCurrentTableGlobal,currentT));
                    denominator = length(excludeCurrentTableGlobal);
                    lambdaforNewDish = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
                    for tempX = xCurrentTable'
                        delta = delta * exp(-lambdaforNewDish)*(lambdaforNewDish^tempX)/factorial(tempX);
                    end
                    pSelectDishes(nDishesT+1,1) = beta(currentT,nDishesT+1) * delta;

                    % sample a dish
                    tempSample = sample(pSelectDishes);

                    % if it is a new dish
                    oldDish = kForCustomers(relevantRows(1),currentT);
                    if tempSample == nDishesT+1
                        newDish = max(allDishesT) + 1;

                        % update beta and numberOfDishesT
                        % insert tempB * betaU, and the new betaU = (1-tempB) * betaU
                        tempB = betarnd(1,gamma);
                        tempBeta = beta(currentT, 1:(nDishesT+1));
                        betaU = tempBeta(end);
                        tempBeta = [tempBeta(1:(end-1)), tempB * betaU, (1-tempB) * betaU];
                        beta(currentT,1:(nDishesT+2)) = tempBeta;

                        numberOfDishesT(currentT,1) = numberOfDishesT(currentT,1) + 1;

                        kForCustomers(relevantRows,currentT) = newDish;

                        % update dish on this table (only update if select another dish)
                    elseif oldDish ~= allDishesT(tempSample)
                        kForCustomers(relevantRows,currentT) = allDishesT(tempSample);
                    end

                    % if there is an empty dish, remove that dish
                    oldDishDisappears = isempty(find(kForCustomers(:,currentT)==oldDish, 1));
                    if oldDishDisappears
                        kToRemove = find(allDishesT(:)==oldDish);
                        betaOldRow = beta(currentT,:);
                        betaOldRow(kToRemove) = [];
                        betaNew = [betaOldRow,-1];
                        beta(currentT,:) = betaNew;
                        numberOfDishesT(currentT,1) = numberOfDishesT(currentT,1) - 1;
                    end
                end
            end
        end

        % Step 3: sampling beta at top level
        beta = -1 * ones(numberOfDays, numberOfMaxDishes);
        for currentT = 1:numberOfDays
            allDishesT = unique(kForCustomers(:,currentT));
            tempNumberOfDishesT = length(allDishesT);
            numberOfGroupsForAllRounds(round, currentT) = tempNumberOfDishesT;

            countTablesMK = zeros(tempNumberOfDishesT,1);
            for dishK = 1:tempNumberOfDishesT
                tempDish = allDishesT(dishK,1);
                for target = selectedTargets
                    relevantRows = find(kForCustomers(:,currentT)==tempDish & sideInfoMatrix(:,1)==target);
                    countTablesMK(dishK,1) = countTablesMK(dishK,1) + length(unique(piCurrent(relevantRows,currentT)));
                end
            end
            beta(currentT,1:(tempNumberOfDishesT+1)) = drchrnd([countTablesMK(:,1)', gamma],1);
        end
    end

    % compute final lambda including all customers using MAP
    lambdaDishTAll = -1 * ones(numberOfDays,numberOfMaxDishes);
    for currentT = 1:numberOfDays
        allDishesT = unique(kForCustomers(:,currentT));
        nAllDishesT = length(allDishesT);

        for index = 1:nAllDishesT
            dish = allDishesT(index,1);
            numerator = sum(dataMatrix(kForCustomers(:,currentT) == dish,currentT));
            denominator = length(dataMatrix(kForCustomers(:,currentT) == dish,currentT));
            lambdaDishTAll(currentT,index) = (numerator + gammaAlpha - 1)/(denominator + 1/gammaBeta);
        end
    end
    
    % save result and return to main
    results.logLikelihood = logLikelihood;
    results.lambdaDishTAll = lambdaDishTAll;
    results.piCurrent = piCurrent;
    results.rhoCurrent = rhoCurrent;

end