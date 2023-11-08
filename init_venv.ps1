function InitVenv {
    python -m venv env
    & "env\Scripts\Activate.ps1"
    pip install -r requirements.txt
    deactivate
}

function friday {
    & "env\Scripts\Activate.ps1"
    python aux.py $($args -join ' ')
    deactivate
}
