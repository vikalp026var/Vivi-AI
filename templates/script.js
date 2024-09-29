document.getElementById('mobile-menu').addEventListener('click', function() {
    const navLinks = document.querySelector('.nav-links');
    const menuToggle = document.getElementById('mobile-menu');
  
    navLinks.classList.toggle('active');
    menuToggle.classList.toggle('active');
  });


//   function toggleAnswer(element) {
//     const answer = element.nextElementSibling;
//     const arrow = element.querySelector('.arrow');
    
//     // Collapse any currently open answer
//     const currentlyOpenAnswer = document.querySelector('.answer[style*="max-height"]');
//     const currentlyOpenArrow = document.querySelector('.arrow.arrow-up');

//     if (currentlyOpenAnswer && currentlyOpenAnswer !== answer) {
//         currentlyOpenAnswer.style.maxHeight = null;
//         currentlyOpenArrow.classList.remove('arrow-up');
//         currentlyOpenArrow.classList.add('arrow-down');
//     }

//     if (answer.style.maxHeight) {
//         // Collapse if the clicked answer is already open
//         answer.style.maxHeight = null;
//         arrow.classList.remove('arrow-up');
//         arrow.classList.add('arrow-down');
//     } else {
//         // Expand the clicked answer
//         answer.style.maxHeight = answer.scrollHeight + 'px';
//         arrow.classList.remove('arrow-down');
//         arrow.classList.add('arrow-up');
//     }
// }

document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting the traditional way

    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    if (file) {
        // Show progress circle
        document.getElementById('progressContainer').style.display = 'block';

        const reader = new FileReader();
        reader.onload = function(e) {
            const displayImage = document.getElementById('displayImage');
            displayImage.src = e.target.result;

            // Simulate a delay for prediction
            setTimeout(() => {
                // Hide progress circle after processing (simulate prediction complete)
                document.getElementById('progressContainer').style.display = 'none';
            }, 3000); // Adjust this timeout to match your processing time
        };
        reader.readAsDataURL(file);
    }
});
const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const imgView = document.getElementById("img-view");

inputFile.addEventListener("change", uploadImage);

function uploadImage() {
    const file = inputFile.files[0];
    if (file) {
        const imgLink = URL.createObjectURL(file);
        imgView.style.backgroundImage = `url(${imgLink})`;
        imgView.style.backgroundSize = 'contain';
        imgView.style.backgroundSize = 'cover'; // Ensure the image covers the container
        imgView.style.backgroundPosition = 'center'; // Center the background image
        imgView.style.width = '80%'; // Adjust width as needed
        imgView.style.height = '300px'; // Set a smaller height
        // Optional: Remove the image URL from memory after usage
        setTimeout(() => URL.revokeObjectURL(imgLink), 1000);
    }
}


function startProcess() {
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    
    const interval = setInterval(() => {
        progress += 10; // Increase progress
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';

        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 1000); // Update every second
}
