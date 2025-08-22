import json
import re

# Read the notebook
with open('ARIMAX models.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the plot_arimax_results function (execution_count = 15)
for i, cell in enumerate(notebook['cells']):
    if (cell.get('cell_type') == 'code' and 
        'execution_count' in cell and 
        cell['execution_count'] == 15):
        
        # Get the source code
        source = ''.join(cell['source'])
        
        # Fix the indentation issue
        # Replace the incorrectly indented comment and forecast generation
        old_pattern = r'        print\(f"AIC: \{model_fit\.aic\} \| BIC: \{model_fit\.bic\}"\)\n    # print\(f"\\\\nBest model: ARIMA\{model\.order\}"\)\n\n        forecast = model_fit\.predict\('
        new_pattern = r'        print(f"AIC: {model_fit.aic} | BIC: {model_fit.bic}")\n        # print(f"\\nBest model: ARIMA{model.order}")\n\n        forecast = model_fit.predict('
        
        source = re.sub(old_pattern, new_pattern, source)
        
        # Fix the title generation to handle both auto and non-auto cases
        old_title_pattern = r'    title = f"\{title_prefix\}:, order \{model\.order\}"'
        new_title_pattern = r'    if auto:\n        title = f"{title_prefix}:, order {model.order}"\n    else:\n        title = f"{title_prefix}:, order {order}"'
        
        source = re.sub(old_title_pattern, new_title_pattern, source)
        
        # Update the cell source
        cell['source'] = source.split('\n')
        # Ensure each line ends with \n except the last one
        for j in range(len(cell['source']) - 1):
            if not cell['source'][j].endswith('\n'):
                cell['source'][j] += '\n'
        
        print(f"Fixed cell {i} (execution_count {cell['execution_count']})")
        break

# Write the fixed notebook
with open('ARIMAX models.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully!")