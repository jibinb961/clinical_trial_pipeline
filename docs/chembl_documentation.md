Certainly! Here's a comprehensive guide to using the **ChEMBL Python client (`chembl_webresource_client`)** to retrieve **modality** and **target** information for drugs.

---

## 🧪 Overview: ChEMBL Python Client

The `chembl_webresource_client` is the official Python library developed by the ChEMBL team. It simplifies access to ChEMBL's RESTful API, allowing you to query data without dealing directly with HTTP requests.

* **Installation**:

```bash
  pip install chembl_webresource_client
```



* **Documentation**: [GitHub Repository](https://github.com/chembl/chembl_webresource_client)

---

## 🔍 Retrieving Drug Modality and Target Information

To obtain modality and target details for a specific drug, follow these steps:

### 1. **Import Necessary Modules**

```python
from chembl_webresource_client.new_client import new_client
```



### 2. **Search for the Drug**

Use the `molecule` resource to search for the drug by name.

```python
molecule = new_client.molecule
results = molecule.filter(pref_name__iexact="aspirin")
```



This returns a list of molecules matching the preferred name "aspirin".([The ChEMBL-og][1])

### 3. **Retrieve Mechanism of Action**

Once you have the ChEMBL ID of the molecule (e.g., `CHEMBL25` for aspirin), you can fetch its mechanism of action.

```python
mechanism = new_client.mechanism
moa = mechanism.filter(molecule_chembl_id="CHEMBL25")
```



This provides details about the drug's mechanism, including the target.

### 4. **Extract Target Information**

From the mechanism data, extract the target ChEMBL ID and retrieve detailed information about the target.

```python
target = new_client.target
target_info = target.filter(target_chembl_id="CHEMBL204")
```



Replace `"CHEMBL204"` with the actual target ID obtained from the mechanism data.

---

## 🧰 Example: Fetching Modality and Target for Aspirin

```python
from chembl_webresource_client.new_client import new_client

# Initialize resources
molecule = new_client.molecule
mechanism = new_client.mechanism
target = new_client.target

# Search for aspirin
results = molecule.filter(pref_name__iexact="aspirin")
if results:
    chembl_id = results[0]['molecule_chembl_id']
    
    # Get mechanism of action
    moa = mechanism.filter(molecule_chembl_id=chembl_id)
    if moa:
        target_id = moa[0]['target_chembl_id']
        
        # Get target information
        target_info = target.filter(target_chembl_id=target_id)
        if target_info:
            print(f"Drug: {results[0]['pref_name']}")
            print(f"Modality: {results[0]['molecule_type']}")
            print(f"Target: {target_info[0]['pref_name']}")
        else:
            print("Target information not found.")
    else:
        print("Mechanism of action not found.")
else:
    print("Drug not found.")
```



This script searches for aspirin, retrieves its modality, and identifies its primary target.

---

## 📘 Additional Resources

* **ChEMBL API Documentation**: [ChEMBL API Docs](https://www.ebi.ac.uk/chembl/api/data/docs)

* **ChEMBL GitHub Repository**: [chembl\_webresource\_client](https://github.com/chembl/chembl_webresource_client)

* **Interactive Examples**: [ChEMBL Jupyter Notebooks](https://github.com/chembl/notebooks/blob/main/ChEMBL_webresource_client_examples.ipynb)([GitHub][2])

---

If you need assistance integrating this into your AI project or have further questions, feel free to ask!

[1]: https://chembl.blogspot.com/2015/02/using-new-chembl-web-services.html?utm_source=chatgpt.com "Using the New ChEMBL Web Services"
[2]: https://github.com/chembl/notebooks/blob/main/ChEMBL_webresource_client_examples.ipynb?utm_source=chatgpt.com "notebooks/ChEMBL_webresource_client_examples.ipynb at main"
