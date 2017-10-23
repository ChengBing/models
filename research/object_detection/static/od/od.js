var image_file	    = document.getElementById('image_file');
var image_info      = document.getElementById('image_info');
var img_a   = document.getElementById('img_a'); 
var img_a4jq   = $('#img_a');
var boxes

function fun_image_preview(){
	// 检查文件是否选择:
	if(!image_file.value){
		image_info.innerHTML = '没有选择文件';
		return;
	}

	// 获取File引用:
	var file = image_file.files[0];
	//判断文件大小
	var size = file.size;
	if(size > 2*1024*1024){
		alert('文件不能大于2MB!');
		return false;
	}
	// 获取File信息:
	image_info.innerHTML = '文件: ' + file.name + '<br>' + '大小: ' + file.size + '<br>';
	if(file.type !== 'image/jpeg' && file.type !== 'image/png' && file.type !== 'image/jpg'){
		alert('不是有效的图片文件!');
		return;
	}
	
	$(document).find("div[name='border']").remove();
	boxes = null;
	
	img_a.src=  getObjectURL(image_file.files[0]); //使用地址
}

function getObjectURL(file) {
    var url = null;
    if (window.createObjectURL != undefined) { // basic
        url = window.createObjectURL(file);
    } else if (window.URL != undefined) { // mozilla(firefox)
        url = window.URL.createObjectURL(file);
    } else if (window.webkitURL != undefined) { // webkit or chrome
        url = window.webkitURL.createObjectURL(file);
    }
    return url;
}

function boxes_view(boxes){
    for(var b in boxes){
        var str = "<div name=\"border\"  style=\"position: absolute; border: 2px solid red;top:$(top);left:$(left);width:$(width);height:$(height);\">$(content)</div>";
        var b1 = (boxes[b]);
        str = str.replace("$(top)", parseInt(img_a.style.top) +  b1.box.ymin * img_a.naturalHeight);
        str = str.replace("$(left)", parseInt(img_a.style.left) + b1.box.xmin * img_a.naturalWidth);
        str = str.replace("$(width)", (b1.box.xmax - b1.box.xmin)* img_a.naturalWidth);
        str = str.replace("$(height)", (b1.box.ymax - b1.box.ymin)* img_a.naturalHeight);
        str = str.replace("$(content)", b1.name + ":" +b1.score.toFixed(2));
        img_a4jq.after(str);
    }
}


$(function () {
    var ajaxFormOption = {
        type: "post", //提交方式
//        dataType: "json", //数据类型
        url: "/od", //请求url
        success: function (data) { //提交成功的回调函数
			console.log(data);
			boxes = JSON.parse(data);
			boxes_view(boxes);
        }
    };

    $("#submit_btn").click(function () {
		if(!image_file.value){
			alert('没有选择文件');
			return;
		}
		$(document).find("div[name='border']").remove();
        $("#image_upload_form").ajaxSubmit(ajaxFormOption);
    });
});

$(function () {
    $("#hide_btn").click(function () {
		if(boxes){
			$(document).find("div[name='border']").remove();
		}
    });
});

$(function () {
    $("#view_btn").click(function () {
		if(boxes){
			boxes_view(boxes);
		}
		else{
		    alert('尚无识别结果');
		}
    });
});


