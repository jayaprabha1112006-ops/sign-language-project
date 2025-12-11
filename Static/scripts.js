
function goHome() {
    window.location.href = "/";
}

function goToStatic() {
    window.location.href = "/static_prediction";
}

function goToDynamic() {
    window.location.href = "/dynamic_prediction";
}

function goToLettersToSigns() {
    window.location.href = "/letters_to_signs";
}

function goToSignsToLetters() {
    window.location.href = "/signs_to_letters";
}
function goToPractice() {
    window.location.href = "/Practice";
}



function showSigns() {
    fetch('/run_static_letters', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(err => alert("Error: " + err));
}

function uploadAndPredict() {
    fetch('/run_static_signs', { method: 'POST' })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(err => alert("Error: " + err));
}

function startDynamic() {
    fetch('/run_dynamic', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerText = data.message;
        })
        .catch(err => alert("Error: " + err));
}


document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const resultDiv = document.getElementById('result');
    const predictBtn = document.getElementById('predictBtn');

   
    if (imageInput) {
        imageInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    }

   
    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
           
            resultDiv.innerText = "Predicting... (connect backend here)";
        });
    }
});

