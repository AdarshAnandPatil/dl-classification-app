document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-input");
    const classifyButton = document.getElementById("classify-btn");
    const imagePreview = document.getElementById("image-preview");
    const previewContainer = document.getElementById("preview-container");
    const loading = document.getElementById("loading");
    const resultMessage = document.getElementById("result-message");
    const resultDetails = document.getElementById("result-details");

    if (!fileInput || !classifyButton) return;

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (!file) {
            previewContainer.style.display = "none";
            return;
        }
        imagePreview.src = URL.createObjectURL(file);
        previewContainer.style.display = "block";
        resultMessage.textContent = "Image selected. Click Classify Image.";
        resultDetails.innerHTML = "";
    });

    // Kept as a named function because older versions of classifier.html
    // may still call classifyImage() from the button.
    window.classifyImage = async function () {
        const file = fileInput.files[0];

        if (!file) {
            alert("Please choose an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        loading.style.display = "block";
        resultMessage.textContent = "Analyzing image...";
        resultDetails.innerHTML = "";
        classifyButton.disabled = true;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const contentType = response.headers.get("content-type") || "";
            const responseText = await response.text();

            if (!contentType.includes("application/json")) {
                throw new Error(
                    "Server returned an invalid response (HTTP " +
                    response.status +
                    "). Please check the Render server logs."
                );
            }

            let data;
            try {
                data = JSON.parse(responseText);
            } catch {
                throw new Error("Server returned invalid JSON.");
            }

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Prediction failed.");
            }

            resultMessage.innerHTML =
                "<strong>Prediction:</strong> " + escapeHtml(data.prediction) +
                "<br><strong>Confidence:</strong> " + data.confidence + "%";

            let html = "<h4>Class Probabilities</h4><ul>";
            for (const [name, probability] of Object.entries(data.probabilities || {})) {
                html += "<li>" + escapeHtml(name) + " : " + probability + "%</li>";
            }
            html += "</ul>";
            resultDetails.innerHTML = html;

        } catch (error) {
            console.error("Prediction error:", error);
            resultMessage.textContent = "❌ " + error.message;
        } finally {
            loading.style.display = "none";
            classifyButton.disabled = false;
        }
    };

    classifyButton.addEventListener("click", window.classifyImage);

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, function (char) {
            return {
                "&": "&amp;", "<": "&lt;", ">": "&gt;",
                '"': "&quot;", "'": "&#039;"
            }[char];
        });
    }
});
