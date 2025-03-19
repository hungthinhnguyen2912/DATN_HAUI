import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:get/get_navigation/src/root/get_material_app.dart';
import 'package:vegetable_classification/App_Color.dart';
import 'package:vegetable_classification/views/auth/splash_screen.dart';

import 'P.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized(); // Đảm bảo Flutter đã được khởi tạo
  await Firebase.initializeApp(); // Khởi tạo Firebase sau khi bindings sẵn sàng
  P.initialController();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(scaffoldBackgroundColor: AppColors.white),
      home: SplashScreen(),
    );
  }
}
