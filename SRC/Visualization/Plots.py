import seaborn as sns
import matplotlib.pyplot as plt



def plot_histograms(train):
    """Plot histograms for various features."""
    plt.hist(train['Tenure'], bins=20, edgecolor='black')
    plt.title('Distribution of Tenure')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Number of Customers')
    plt.show()
    # distribution of Age
    plt.hist(train['Age'], bins=20, edgecolor='black')
    plt.title('Distribution of Age')            
    plt.xlabel('Age (years)')
    plt.ylabel('Number of Customers')
    plt.show()
    # distribution of Payment Delay
    plt.hist(train['Payment Delay'], bins=20, edgecolor='black')
    plt.title('Distribution of Payment Delay')            
    plt.xlabel('Payment Delay (days)')
    plt.ylabel('Number of Customers')
    plt.show()
    # distribution of Last Interaction
    plt.hist(train['Last Interaction'], bins=20, edgecolor='black')
    plt.title('Distribution of Last Interaction')            
    plt.xlabel('Last Interaction (days)')
    plt.ylabel('Number of Customers')
    plt.show()
    # distribution of Support Calls
    plt.hist(train['Support Calls'], bins=20, edgecolor='black')
    plt.title('Distribution of Support Calls')
    plt.xlabel('Number of Support Calls')
    plt.ylabel('Number of Customers')
    plt.show()
    # countplot of contract length
    sns.countplot(x='Contract Length', data=train)
    plt.title('Contract Length Distribution')
    plt.xlabel('Contract Length')
    plt.ylabel('Number of Customers')
    plt.show()
    
