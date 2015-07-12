""" Heuristics used to query the most uncertain candidate out of the unlabelled pool. """


def random_h(X_training_candidates, **kwargs):
    """ Return a random candidate. """
    
    random_index = np.random.choice(np.arange(0, len(X_training_candidates)), 1, replace=False)
    return random_index


def entropy_h(X_training_candidates, **kwargs):
    """ Return the candidate whose prediction vector displays the greatest Shannon entropy. """
    
    # get the classifier
    classifier = kwargs['classifier']
    
    # predict probabilities
    probs = classifier.predict_proba(X_training_candidates)
    
    # comptue Shannon entropy
    shannon = -np.sum(probs * np.log(probs), axis=1)
    
    # pick the candidate with the greatest Shannon entropy
    greatest_shannon = np.argmax(shannon)
    
    return [greatest_shannon]


def margin_h(X_training_candidates, **kwargs):
    """ Return the candidate with the smallest margin.
    
        The margin is defined as the difference between the two largest values
        in the prediction vector.
    """
    
    # get the classifier
    classifier = kwargs['classifier']
    
    # predict probabilities
    probs = classifier.predict_proba(X_training_candidates)
    
    # sort the probabilities from smallest to largest
    probs = np.sort(probs, axis=1)
    
    # compute the margin (difference between two largest values)
    margins = np.abs(probs[:,-1] - probs[:,-2])
    
    # pick the candidate with the smallest margin
    smallest_margin = np.argmin(margins)
    
    return [smallest_margin]


def qbb_margin_h(X_training_candidates, **kwargs):
    """ Return the candidate with the smallest average margin.
    
        We first use bagging to train k classifiers. The margin is then defined as
        the average difference between the two largest values in the prediction vector.
    """
    
    # extract parameters
    committee = kwargs['committee']
    bag_size = kwargs['bag_size']
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    n_classes = kwargs['n_classes']
    
    # intialise probability matrix
    probs = np.zeros((len(X_training_candidates), n_classes))
    
    # train each member of the committee
    for member in committee:
        
        # randomly select a bag of samples
        member_train_index = np.random.choice(np.arange(0, len(y_train)), bag_size, replace=True)
        member_X_train = X_train[member_train_index]
        member_y_train = y_train[member_train_index]
        
        # train member and predict
        member.fit(member_X_train, member_y_train)
        prob = member.predict_proba(X_training_candidates)
        
        # make sure all class predictions are present
        prob_full = np.zeros((prob.shape[0], n_classes))
        
        # case 1: only galaxy predictions are generated
        if prob.shape[1] == 1:
            prob_full[:,0] += prob
            
        # case 2: only galaxy and star predictions are generated
        if prob.shape[1] == 2:
            prob_full[:,0] += prob[:,0]
            prob_full[:,2] += prob[:,1]
            
        # case 3: all class predictions are generated
        else:
            prob_full += prob
            
        # accumulate probabilities
        probs += prob_full
            
    # average out the probabilities
    probs /= len(committee)
    
    # sort the probabilities from smallest to largest
    probs = np.sort(probs, axis=1)
    
    # compute the margin (difference between two largest values)
    margins = np.abs(probs[:,-1] - probs[:,-2])

    # pick the candidate with the smallest margin
    smallest_margin = np.argmin(margins)
    
    return [smallest_margin]


def qbb_kl_h(X_training_candidates, **kwargs):
    """ Return the candidate with the largest average KL divergence from the mean.
    
        We first use bagging to train k classifiers. We then choose the candidate
        that has the largest Kullback–Leibler divergence from the average
    """
    
    # extract parameters
    committee = kwargs['committee']
    bag_size = kwargs['bag_size']
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    n_classes = kwargs['n_classes']
    
    # intialise probability matrix
    avg_probs = np.zeros((len(X_training_candidates), n_classes))
    probs = []
    
    # train each member of the committee
    for member in committee:
        
        # randomly select a bag of samples
        member_train_index = np.random.choice(np.arange(0, len(y_train)), bag_size, replace=True)
        member_X_train = X_train[member_train_index]
        member_y_train = y_train[member_train_index]
        
        # train member and predict
        member.fit(member_X_train, member_y_train)
        prob = member.predict_proba(X_training_candidates)
        
        # make sure all class predictions are present
        prob_full = np.zeros((prob.shape[0], n_classes))
        
        # case 1: only galaxy predictions are generated
        if prob.shape[1] == 1:
            prob_full[:,0] += prob
            
        # case 2: only galaxy and star predictions are generated
        if prob.shape[1] == 2:
            prob_full[:,0] += prob[:,0]
            prob_full[:,2] += prob[:,1]
            
        # case 3: all class predictions are generated
        else:
            prob_full += prob
            
        # accumulate probabilities
        probs.append(prob_full)
        avg_probs += probs[-1]
        
    # average out the probabilities
    avg_probs /= len(committee)
    
    # compute the KL divergence
    avg_kl = np.zeros(avg_probs.shape[0])
    for p in probs:
        kl = np.sum(p * np.log(p / avg_probs), axis=1)
        avg_kl += kl
    
    # average out the KL divergence
    avg_kl /= len(committee)
    
    # extract the candidate with the largest average divergence
    largest_kl = np.argmax(avg_kl)
    
    return [largest_kl]