import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/auth_email_and_password/register_page.dart';

import '../../App_Color.dart';
import 'auth_email_and_password/log_in_page.dart';

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
          elevation: 0,
          bottom: TabBar(
            tabs: [
              Tab(text: "Log In"),
              Tab(text: "Register"),
            ],
            labelColor: AppColors.green,
            unselectedLabelColor: Colors.grey,
            indicatorColor: AppColors.black,
            indicatorWeight: 3,
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
