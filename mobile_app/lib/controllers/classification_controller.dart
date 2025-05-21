import 'dart:io';

import 'package:get/get.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import '../P.dart';
import '../models/History.dart';

class ClassificationController extends GetxController {
  // List<String> label = [
  //   'Amaranth',
  //   'Apple 10',
  //   'Apple 12',
  //   'Apple 13',
  //   'Apple 14',
  //   'Apple 6',
  //   'Apple 9',
  //   'Apple Braeburn 1',
  //   'Apple Crimson Snow 1',
  //   'Apple Golden 1',
  //   'Apple Golden 2',
  //   'Apple Golden 3',
  //   'Apple Granny Smith 1',
  //   'Apple hit 1',
  //   'Apple Pink Lady 1',
  //   'Apple Red 1',
  //   'Apple Red 2',
  //   'Apple Red 3',
  //   'Apple Red Delicious 1',
  //   'Apple Red Yellow 1',
  //   'Apple Red Yellow 2',
  //   'Apple Rotten 1',
  //   'Apricot 1',
  //   'Avocado 1',
  //   'Avocado ripe 1',
  //   'Banana 1',
  //   'Banana 3',
  //   'Banana Lady Finger 1',
  //   'Banana Red 1',
  //   'Beans 1',
  //   'Beetroot 1',
  //   'Bitter Gourd',
  //   'Blackberrie 1',
  //   'Blackberrie 2',
  //   'Blackberrie half rippen 1',
  //   'Blackberrie not rippen 1',
  //   'Blueberry 1',
  //   'Bottle Gourd',
  //   'Broccoli',
  //   'Cabbage red 1',
  //   'Cabbage white 1',
  //   'Cactus fruit 1',
  //   'Cactus fruit green 1',
  //   'Cactus fruit red 1',
  //   'Cantaloupe 1',
  //   'Cantaloupe 2',
  //   'Carambula 1',
  //   'Carrot 1',
  //   'Cauliflower 1',
  //   'Cherry 1',
  //   'Cherry 2',
  //   'Cherry Rainier 1',
  //   'Cherry Wax Black 1',
  //   'Cherry Wax not rippen 1',
  //   'Cherry Wax Red 1',
  //   'Cherry Wax Yellow 1',
  //   'Chestnut 1',
  //   'Clementine 1',
  //   'Cocos 1',
  //   'Corn 1',
  //   'Corn Husk 1',
  //   'Cucumber 1',
  //   'Cucumber 10',
  //   'Cucumber 3',
  //   'Cucumber 9',
  //   'Cucumber Ripe 1',
  //   'Cucumber Ripe 2',
  //   'Dates 1',
  //   'Eggplant 1',
  //   'Eggplant long 1',
  //   'Fig 1',
  //   'Garlic',
  //   'Ginger Root 1',
  //   'Gooseberry 1',
  //   'Granadilla 1',
  //   'Grape Blue 1',
  //   'Grape Pink 1',
  //   'Grape White 1',
  //   'Grape White 2',
  //   'Grape White 3',
  //   'Grape White 4',
  //   'Grapefruit Pink 1',
  //   'Grapefruit White 1',
  //   'Guava 1',
  //   'Hazelnut 1',
  //   'Huckleberry 1',
  //   'Kaki 1',
  //   'Kiwi 1',
  //   'Kohlrabi 1',
  //   'Kumquats 1',
  //   'Lemon 1',
  //   'Lemon Meyer 1',
  //   'Limes 1',
  //   'Lychee 1',
  //   'Mandarine 1',
  //   'Mango 1',
  //   'Mango Red 1',
  //   'Mangostan 1',
  //   'Maracuja 1',
  //   'Melon Piel de Sapo 1',
  //   'Mulberry 1',
  //   'Nectarine 1',
  //   'Nectarine Flat 1',
  //   'Nut Forest 1',
  //   'Nut Pecan 1',
  //   'Okra',
  //   'Onion Red 1',
  //   'Onion Red Peeled 1',
  //   'Onion White 1',
  //   'Orange 1',
  //   'Papaya 1',
  //   'Passion Fruit 1',
  //   'Peach 1',
  //   'Peach 2',
  //   'Peach Flat 1',
  //   'Pear 1',
  //   'Pear 2',
  //   'Pear 3',
  //   'Pear Abate 1',
  //   'Pear Forelle 1',
  //   'Pear Kaiser 1',
  //   'Pear Monster 1',
  //   'Pear Red 1',
  //   'Pear Stone 1',
  //   'Pear Williams 1',
  //   'Pepino 1',
  //   'Pepper Green 1',
  //   'Pepper Orange 1',
  //   'Pepper Red 1',
  //   'Pepper Yellow 1',
  //   'Physalis 1',
  //   'Physalis with Husk 1',
  //   'Pineapple 1',
  //   'Pineapple Mini 1',
  //   'Pistachio 1',
  //   'Pitahaya Red 1',
  //   'Plum 1',
  //   'Plum 2',
  //   'Plum 3',
  //   'Pomegranate 1',
  //   'Pomelo Sweetie 1',
  //   'Potato Red 1',
  //   'Potato Red Washed 1',
  //   'Potato Sweet 1',
  //   'Potato White 1',
  //   'Pumpkin',
  //   'Quince 1',
  //   'Radish',
  //   'Rambutan 1',
  //   'Raspberry 1',
  //   'Redcurrant 1',
  //   'Salak 1',
  //   'Spinach',
  //   'Strawberry 1',
  //   'Strawberry Wedge 1',
  //   'Tamarillo 1',
  //   'Tangelo 1',
  //   'Tomato 1',
  //   'Tomato 2',
  //   'Tomato 3',
  //   'Tomato 4',
  //   'Tomato Cherry Red 1',
  //   'Tomato Heart 1',
  //   'Tomato Maroon 1',
  //   'Tomato not Ripened 1',
  //   'Tomato Yellow 1',
  //   'Walnut 1',
  //   'Watermelon 1',
  //   'Zucchini 1',
  //   'Zucchini dark 1',
  // ];
  List<String> label = [
    'Bean',
    'Bitter_Gourd',
    'Bottle_Gourd',
    'Brinjal',
    'Broccoli',
    'Cabbage',
    'Capsicum',
    'Carrot',
    'Cauliflower',
    'Cucumber',
    'Papaya',
    'Potato',
    'Pumpkin',
    'Radish',
    'Tomato',
  ];
  late Interpreter _interpreter;
  RxString result = "No classification yet".obs;

  @override
  void onInit() {
    // TODO: implement onInit
    super.onInit();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      print("Đang tải mô hình từ assets...");
      final options = InterpreterOptions();
      _interpreter = await Interpreter.fromAsset(
        'assets/tflite/xInception.tflite',
        options: options,
      );
      print("Mô hình local đã tải thành công!");
    } catch (e) {
      print("Lỗi tải mô hình local: $e");
    }
    print("Input tensor shape: ${_interpreter.getInputTensor(0).shape}");
    print("Output tensor shape: ${_interpreter.getOutputTensor(0).shape}");
  }

  Future<void> classifyImage(File imageFile) async {
    final stopwatch = Stopwatch()..start();
    img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
    if (image == null) {
      result.value = "Error loading image";
      return;
    }

    image = img.copyResize(image, width: 224, height: 224);
    List input = List.generate(
      1,
          (i) => List.generate(
        224,
            (y) => List.generate(224, (x) => List.filled(3, 0.0)),
      ),
    );

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        var pixel = image.getPixelSafe(x, y);
        if (pixel is img.PixelUint8) {
          num red = pixel.r;
          num green = pixel.g;
          num blue = pixel.b;

          input[0][y][x][0] = red.toDouble() /255.0;
          input[0][y][x][1] = green.toDouble() / 255.0;
          input[0][y][x][2] = blue.toDouble() / 255.0;
        }
      }
    }
    List output = List.generate(1, (i) => List.filled(15, 0.0));
    _interpreter.run(input, output);
    stopwatch.stop();
    print("Classification time: ${stopwatch.elapsedMilliseconds} ms");
    List<double> probabilities = output[0].cast<double>();
    int labelIndex = probabilities.indexOf(
      probabilities.reduce((a, b) => a > b ? a : b),
    );
    result.value = label[labelIndex];
  }

  Future<void> postHistory(History his) async {
    P.fireStore.collection("History").add(his.toJson());
  }

  void clearResult() {
    result.value = "No classification yet";
  }
}


