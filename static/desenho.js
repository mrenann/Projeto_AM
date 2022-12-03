var canvas;
var context;
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint = false;
var curColor = "#000";


window.onload = function () {

    canvas = document.getElementById('canvas');
    context = document.getElementById('canvas').getContext("2d");

    canvas.width = window.innerHeight*0.65;
    canvas.height = window.innerHeight*0.65;

    canvas.onmousedown = function (e) {
        var mouseX = e.pageX - this.offsetLeft;
        var mouseY = e.pageY - this.offsetTop;

        paint = true;
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
        redraw();
    };

    canvas.onmousemove = function (e) {
        if (paint) {
            addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
            redraw();
        }
    };

    canvas.onmouseup = function (e) {
        paint = false;
    };
}


function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

function abc(){
const canvas = document.getElementById('canvas')
const img    = canvas.toDataURL('image/png')
    document.write('<p>SEU DESENHO!</p>' +
        '<img src="'+img+'"/>' +
        '');
}

function redraw() {

    context.clearRect(0,0, canvas.width, canvas.height);
    context.strokeStyle = curColor;
    context.lineJoin = "round";
    context.lineWidth = 3;
for (var i = 0; i < clickX.length; i++) {
    context.beginPath();
    if (clickDrag[i] && i) {
        context.moveTo(clickX[i - 1], clickY[i - 1]);
    } else {
        context.moveTo(clickX[i] - 1, clickY[i]);
    }
    context.lineTo(clickX[i], clickY[i]);
    context.closePath();
    context.stroke();
}
}