import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';

import 'package:get/get_navigation/src/root/get_material_app.dart';
import 'package:mobile_app/views/auth/splash_screen.dart';
import 'App_Color.dart';
import 'P.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  P.initialController();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.white,
        primaryColor: AppColors.green,
        secondaryHeaderColor: AppColors.green,
      ),
      home: SplashScreen(),
    );
  }
}
