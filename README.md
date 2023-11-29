<h1> The Impact of Public Income Tax Return Disclosure on Tax Avoidance and Tax Evasion – Insights from an Agent-Based Model</h1>

<p>This repository contains python code for the agent based simulation described in the above paper (published in FinanzArchiv/Public Finance Analysis 79(3), pp. 235–274, 2023).</p>

<p>Notes:</p>
<ul>
	<li>Step-by-step development of the model as well as analysis of the different Scenarios (Section 4.1, Figures 4–6 in the paper) is contained in the file 1_code/TaxReturnDisclosure.ipynb. This file also contains code for analysis of network properties and a figure of the network.</li>
	<li>Code for simulating for many different parameter combinations (Section 4.2 in the paper) with multiprocessing support is contained in the file 1_code/TRDisclosure.py</li>
	<li>The folder 2_temp_files containes simlation results.</li>
	<li>The file 1_code/Data_Analysis.ipynb contains code for OLS regressions over the simulation results (it uses the files in 2_temp_files).</li>
	<li>The folder 3_output contains svg-pictures for Scenarios 1–3, as shown in the paper (produced by TaxReturnDisclosure.ipynb), as well as a Watts-Strogatz network with the parameters used in the paper in the gexf format.</li>
</ul> 

<p>Please feel free to use or modify the code or take any parts of it that might be useful. If you have any questions or comments, please feel free to contact us. Also, we are interested in further research in this area. So if you use the model, modifications of it (or your own model), please drop us a line.</p>

<p>–The authors</p>
