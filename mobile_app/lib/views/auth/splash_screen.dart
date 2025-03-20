import 'package:flutter/material.dart';
import 'package:get/get.dart';


import '../../App_Color.dart';
import 'auth_screen.dart';


class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(Duration(seconds: 3), () {
      Get.off(AuthScreen());
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.green,
      body: Center(
        child: Image.asset(
          'assets/splash_logo.png',
          width: 150,
          height: 150,
        ),
      ),
    );
  }
}
