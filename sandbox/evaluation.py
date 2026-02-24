from itpt.models import get_list, get_model
from itpt.core import parse_newick_string, compare_newick_phylogeny, compare_newick_topology

def run_evaluation():
    newick_string1 = "(leaf:1.0,(leaf:0.6596306068601583,((((leaf:0.28430079155672816,(leaf:0.05408970976253291,leaf:0.05408970976253291):0.23021108179419525):0.16754617414248035,leaf:0.45184696569920846):0.11279683377308707,(leaf:0.3825857519788919,leaf:0.3825857519788919):0.18205804749340368):0.039577836411609495,(leaf:0.5191292875989446,leaf:0.5191292875989446):0.08509234828496041):0.055408970976253295):0.34036939313984166);"
    newick1 = parse_newick_string(newick_string1)

    newick_string2 = "(leaf:1.0,(leaf:0.6596306068601585,(leaf:0.6042216358839052,leaf:0.6042216358839052):0.055408970976253274):0.34036939313984155);"
    newick2 = parse_newick_string(newick_string2)

    metrics = compare_newick_phylogeny(newick1, newick2)
    print(metrics)

    metrics = compare_newick_topology(newick1, newick2)
    print(metrics)
