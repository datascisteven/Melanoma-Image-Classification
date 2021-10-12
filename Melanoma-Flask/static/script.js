var file = document.getElementById("image");

file.onchange = function(){
    if(file.files.length > 0)
    {
      document.getElementById('file-name').innerHTML = file.files[0].name;
    }
};