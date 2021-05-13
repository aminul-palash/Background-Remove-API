$(".fileupload input").change(function() {
    var id = $(this)[0].id;
    var fileInput = document.getElementById("fileInput");
    filename = '';
    if ('files' in fileInput) {
        if (fileInput.files.length == 0) {
            $("#alert-body").text("Please browse for one or more files.");
            $('#alert').modal('show');
            return false;
        }
        else {
            var file = fileInput.files[0];
            if ('name' in file) {
                filename = file.name;
            }
            else {
                filename = file.fileName;
            }
            var displayImg = document.getElementById("photo");
            displayImg.src = URL.createObjectURL(fileInput.files[0]);


                }
            }
});

function Start() {
    var formData = new FormData($("#fileupload")[0]);
    var width = document.getElementById("width").value;
    if (width == "")
    {
        alert("Enter width value!");
        return;
    }
    var height = document.getElementById("height").value;
    if (height == "")
    {
        alert("Enter height value!");
        return;
    }
    var red = document.getElementById("red").value;
    if (red == "")
    {
        alert("Enter red value!");
        return;
    }
    var green = document.getElementById("green").value;
    if (green == "")
    {
        alert("Enter green value!");
        return;
    }
    var blue = document.getElementById("blue").value;
    if (blue == "")
    {
        alert("Enter blue value!");
        return;
    }
    // formData.append("width", width);
    // formData.append("height", height);
    // formData.append("red", red);
    // formData.append("green", green);
    // formData.append("blue", blue);
    // 46,155,236 
    formData.append("width", "55");
    formData.append("height", "55");
    formData.append("red", "255");
    formData.append("green", "255");
    formData.append("blue", "255");

    $.ajax({
        url: "/start",
        type: 'POST',
        data: formData,
        crossDomain: true,
        async: true,
        success: function (data) {
            document.getElementById("result").src = data;
        },
    cache: false,
    contentType: false,
    processData: false
    });
}
