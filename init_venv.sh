function activate_venv(){
    chmod +x runner.py 
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    if [ $1 != "--activate" ]; then
        deactivate
    fi
}

function friday(){
    source env/bin/activate
    python runner.py $@
    deactivate
}
