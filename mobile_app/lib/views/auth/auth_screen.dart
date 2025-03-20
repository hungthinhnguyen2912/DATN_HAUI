import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/register_page.dart';

import '../../App_Color.dart';
import 'log_in_page.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: AppColors.white,
          elevation: 0, // Bỏ bóng để giao diện sạch hơn
          title: Text("Authentication", style: TextStyle(color: AppColors.black, fontWeight: FontWeight.bold)),
          centerTitle: true,
          bottom: TabBar(
            tabs: [
              Tab(text: "Log In"),
              Tab(text: "Register"),
            ],
            labelColor: AppColors.green, // Màu chữ khi được chọn
            unselectedLabelColor: Colors.grey, // Màu chữ khi chưa chọn
            indicatorColor: AppColors.black, // Màu gạch chân tab
            indicatorWeight: 3, // Độ dày của indicator
          ),
        ),
        body: TabBarView(
          children: [
            LogInPage(),
            RegisterPage(),
          ],
        ),
      ),
    );
  }
}
