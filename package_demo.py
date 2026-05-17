import os
import zipfile
import datetime

def package():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('outputs', exist_ok=True)
    zip_path = f"outputs/demo_bundle_{timestamp}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists('app/streamlit_demo.py'):
            zipf.write('app/streamlit_demo.py')
        
        for root, _, files in os.walk('src/req_ambiguity'):
            for file in files:
                if file.endswith('.py'):
                    zipf.write(os.path.join(root, file))
                    
        for root, _, files in os.walk('configs'):
            for file in files:
                if file.endswith('.yaml'):
                    zipf.write(os.path.join(root, file))
                    
        if os.path.exists('outputs/checkpoints/best_model.pt'):
            zipf.write('outputs/checkpoints/best_model.pt')
        if os.path.exists('outputs/results/optimal_thresholds.json'):
            zipf.write('outputs/results/optimal_thresholds.json')
            
        if os.path.exists('requirements.txt'):
            zipf.write('requirements.txt')
            
        if os.path.exists('CHANGES.md'):
            zipf.write('CHANGES.md', 'README.md')
        
    print(f"Created bundle: {zip_path}")

if __name__ == '__main__':
    package()
