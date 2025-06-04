
## Description
Analyzis a snapshot taken of the Warframe Market website. 

## Files
**orders.csv , users.csv** - where the data of the snapshot is stored, if either of them is missing then a new snapshot is taken (**WARNING**: due to API rate limits it will take 20 minutes for a new snapshot) and graph and partition are recalculated

**project.ipynb** - jupyter notebook, it can't run streamlit stuff so must export it to python script before running it

**script.py** - python script resulted after exporting the jupyter notebook, run with "streamlit run script.py"

**linuxEnv.yml, windowsEnv.yml** - conda environments for linux and for windows

**bipartite_graph.pkl** - resulted NetworkX graph, nodes are items and users, and edges are created if the user has an order involving the item; delete it if you want it to be remade

**partition.json** - contains the result of community detection, delete it if you want to redo community detection on another snapshot; delete it if you want it to be remade

## Additional notes
When it says median_platinum_sell and median_platinum_buy it is not true median, it is the median of the five cheapest sell orders and most expensive buy orders made by users active within the last week.
