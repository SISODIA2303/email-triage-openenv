import random
from typing import List
from datetime import datetime, timedelta
from env.models import Email, EmailCategory, Priority, RoutingTeam

#Ground truth labels for every email id
EMAIL_GROUND_TRUTH: dict={}

#Seed for reproducibility
random.seed(42)

BILLING_EMAILS=[
    {
        "subject":"Invoice #4521 overdue - immediate action required",
        "body":"Dear team, our invoice #4521 for $2,400 issued on Oct 1st is now 30 days overdue. Please process payment or contact us to discuss. This may affect service continuity.",
        "category":EmailCategory.BILLING,
        "priority":Priority.P1,
        "routing_team":RoutingTeam.BILLING_TEAM,
        "reply_keywords":["apologize","invoice","resolve","contact"],
    },
    {
        "subject":"Question about my subscription renewal",
        "body": "Hi, I was charged $99 last week but I thought I cancelled my subscription in September. Can you check my account and refund if this was an error? My account email is user@example.com.",
        "category": EmailCategory.BILLING,
        "priority": Priority.P2,
        "routing_team": RoutingTeam.BILLING_TEAM,
        "reply_keywords": ["check", "account", "refund", "subscription", "confirm"],
    },
    {
        "subject": "Request for updated billing address",
        "body": "Hello, we have moved offices. Please update our billing address to: 123 New Street, Suite 400, Chicago IL 60601. Our account number is ACC-8821.",
        "category": EmailCategory.BILLING,
        "priority": Priority.P3,
        "routing_team": RoutingTeam.BILLING_TEAM,
        "reply_keywords": ["updated", "address", "account", "confirm"],
    },
    {
        "subject": "Need a copy of last 3 invoices for audit",
        "body": "Our finance team is conducting an annual audit and requires copies of invoices from January to March 2024. Could you email those to finance@ourcompany.com? Thank you.",
        "category": EmailCategory.BILLING,
        "priority": Priority.P3,
        "routing_team": RoutingTeam.BILLING_TEAM,
        "reply_keywords": ["invoices", "audit", "send", "confirm"],
    },
    {
        "subject": "Payment failed - credit card declined",
        "body": "We attempted to charge your card on file ending in 4242 but the payment was declined. Please update your payment method within 48 hours to avoid service interruption.",
        "category": EmailCategory.BILLING,
        "priority": Priority.P1,
        "routing_team": RoutingTeam.BILLING_TEAM,
        "reply_keywords": ["payment", "update", "card", "urgent", "resolve"],
    },
]

SUPPORT_EMAILS = [
    {
        "subject": "Application crashes on startup after latest update",
        "body": "Since installing version 3.2.1 yesterday, the app crashes immediately on launch. I am on Windows 11, i7 processor, 16GB RAM. Error log attached. This is blocking my entire team of 12 people.",
        "category": EmailCategory.SUPPORT,
        "priority": Priority.P1,
        "routing_team": RoutingTeam.SUPPORT_TEAM,
        "reply_keywords": ["crash", "investigate", "fix", "workaround", "priority"],
    },
    {
        "subject": "Cannot reset my password",
        "body": "I have been trying to reset my password for two days. The reset email never arrives, even after checking spam. My account is john.doe@company.com. Please help.",
        "category": EmailCategory.SUPPORT,
        "priority": Priority.P2,
        "routing_team": RoutingTeam.SUPPORT_TEAM,
        "reply_keywords": ["reset", "password", "email", "account", "assist"],
    },
    {
        "subject": "How do I export data to CSV?",
        "body": "Hi support, I have been looking through the documentation but cannot figure out how to export my project data to CSV format. Is this feature available? If so, where do I find it?",
        "category": EmailCategory.SUPPORT,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.SUPPORT_TEAM,
        "reply_keywords": ["export", "csv", "steps", "guide", "feature"],
    },
    {
        "subject": "Integration with Slack is broken",
        "body": "Our Slack integration stopped posting notifications three days ago. We rely on this for incident alerts. Team of 50 is affected. Please treat this as high priority.",
        "category": EmailCategory.SUPPORT,
        "priority": Priority.P1,
        "routing_team": RoutingTeam.SUPPORT_TEAM,
        "reply_keywords": ["slack", "integration", "fix", "notifications", "investigate"],
    },
    {
        "subject": "Dashboard loading very slowly",
        "body": "For the past week our main analytics dashboard takes over 2 minutes to load. It used to load in under 5 seconds. Nothing has changed on our end. Please look into this.",
        "category": EmailCategory.SUPPORT,
        "priority": Priority.P2,
        "routing_team": RoutingTeam.SUPPORT_TEAM,
        "reply_keywords": ["performance", "dashboard", "investigate", "optimize"],
    },
]

SPAM_EMAILS = [
    {
        "subject": "You have WON $1,000,000 - Claim NOW!!!",
        "body": "Congratulations! You have been selected as our lucky winner. Click here immediately to claim your $1,000,000 prize. Offer expires in 24 hours. Act now!!!",
        "category": EmailCategory.SPAM,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.SPAM_FILTER,
        "reply_keywords": [],
    },
    {
        "subject": "Cheap Rx meds - no prescription needed",
        "body": "Buy cheap medications online without a prescription. Viagra, Cialis, and more. Discreet shipping worldwide. Visit our website now for special discounts.",
        "category": EmailCategory.SPAM,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.SPAM_FILTER,
        "reply_keywords": [],
    },
    {
        "subject": "Your account has been compromised - verify now",
        "body": "We detected suspicious activity on your account. Click the link below immediately to verify your identity and secure your account. Failure to act will result in permanent suspension.",
        "category": EmailCategory.SPAM,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.SPAM_FILTER,
        "reply_keywords": [],
    },
    {
        "subject": "Make $5000/week working from home!!!",
        "body": "Tired of your 9-to-5? Our proven system lets anyone make $5000 per week from home. No experience needed. Join 10,000 happy members today. Limited spots available!",
        "category": EmailCategory.SPAM,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.SPAM_FILTER,
        "reply_keywords": [],
    },
]

SALES_EMAILS = [
    {
        "subject": "Partnership opportunity - 50,000 user base interested in your product",
        "body": "Hi, I represent a network of 50,000 SMB owners who would be a perfect fit for your product. I would love to discuss a referral partnership. Are you available for a 20-minute call this week?",
        "category": EmailCategory.SALES,
        "priority": Priority.P2,
        "routing_team": RoutingTeam.SALES_TEAM,
        "reply_keywords": ["partnership", "discuss", "call", "interested"],
    },
    {
        "subject": "Enterprise plan inquiry for 500 seat license",
        "body": "We are evaluating your platform for our organization of 500 employees. Could you send over enterprise pricing and whether you offer SSO, audit logs, and dedicated support? We need to decide by end of month.",
        "category": EmailCategory.SALES,
        "priority": Priority.P1,
        "routing_team": RoutingTeam.SALES_TEAM,
        "reply_keywords": ["enterprise", "pricing", "demo", "features", "contact"],
    },
    {
        "subject": "Can I get a demo of your product?",
        "body": "Hello, I came across your product on LinkedIn and it looks interesting for our team. Could we schedule a demo sometime next week? We are a 20-person startup in the fintech space.",
        "category": EmailCategory.SALES,
        "priority": Priority.P3,
        "routing_team": RoutingTeam.SALES_TEAM,
        "reply_keywords": ["demo", "schedule", "happy", "team"],
    },
    {
        "subject": "Renewal decision - need 20% discount to stay",
        "body": "Our annual contract is up for renewal next week. We are happy with the product but our budget was cut. We need at least 20% off to continue. Please let me know if this is possible.",
        "category": EmailCategory.SALES,
        "priority": Priority.P1,
        "routing_team": RoutingTeam.SALES_TEAM,
        "reply_keywords": ["renewal", "discount", "discuss", "value", "retain"],
    },
]

HR_EMAILS = [
    {
        "subject": "Harassment complaint - requires immediate attention",
        "body": "I am writing to formally report repeated inappropriate comments made by my manager over the past two weeks. I have documented three separate incidents. I would like to speak with HR confidentially as soon as possible.",
        "category": EmailCategory.HR,
        "priority": Priority.P1,
        "routing_team": RoutingTeam.HR_TEAM,
        "reply_keywords": ["confidential", "seriously", "investigate", "contact", "support"],
    },
    {
        "subject": "Maternity leave request - starting December 1st",
        "body": "Hi HR, I would like to formally request maternity leave starting December 1st for 16 weeks as per company policy. Please let me know what forms I need to complete and who to notify.",
        "category": EmailCategory.HR,
        "priority": Priority.P2,
        "routing_team": RoutingTeam.HR_TEAM,
        "reply_keywords": ["maternity", "forms", "policy", "confirm", "process"],
    },
    {
        "subject": "Question about work from home policy",
        "body": "Hi, I wanted to clarify our current WFH policy. My team lead mentioned it changed last month but I have not received any official communication. Can you share the updated policy document?",
        "category": EmailCategory.HR,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.HR_TEAM,
        "reply_keywords": ["policy", "wfh", "document", "updated", "share"],
    },
    {
        "subject": "Resignation letter - effective two weeks from today",
        "body": "Dear HR, please accept this email as my formal resignation from my position as Senior Developer, effective two weeks from today. It has been a pleasure working here. I am happy to assist with the transition.",
        "category": EmailCategory.HR,
        "priority": Priority.P2,
        "routing_team": RoutingTeam.HR_TEAM,
        "reply_keywords": ["resignation", "acknowledge", "transition", "offboarding"],
    },
]

GENERAL_EMAILS = [
    {
        "subject": "Office kitchen cleanup reminder",
        "body": "Hi everyone, just a friendly reminder to please clean up after yourself in the kitchen. Dishes left in the sink will be removed after 24 hours. Thanks for keeping our space tidy.",
        "category": EmailCategory.GENERAL,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.GENERAL_TEAM,
        "reply_keywords": ["noted", "reminder", "thank"],
    },
    {
        "subject": "Company all-hands meeting - Friday 3pm",
        "body": "Please join us for our quarterly all-hands this Friday at 3pm in the main conference room. CEO will share Q3 results and roadmap for Q4. Attendance is strongly encouraged. Zoom link also available.",
        "category": EmailCategory.GENERAL,
        "priority": Priority.P3,
        "routing_team": RoutingTeam.GENERAL_TEAM,
        "reply_keywords": ["confirm", "attend", "noted"],
    },
    {
        "subject": "New office wifi password",
        "body": "The office wifi password was updated this morning. The new password is: OfficeNet2024! Please update your devices. Contact IT if you have any trouble connecting.",
        "category": EmailCategory.GENERAL,
        "priority": Priority.P4,
        "routing_team": RoutingTeam.GENERAL_TEAM,
        "reply_keywords": ["password", "updated", "contact", "it"],
    },
]

def generate_timestamp(days_ago: int = 0) -> str:
    base = datetime(2024, 11, 1, 9, 0, 0)
    delta = timedelta(days=days_ago, hours=random.randint(0, 8), minutes=random.randint(0, 59))
    return (base - delta).strftime("%Y-%m-%dT%H:%M:%S")


def generate_email_dataset() -> List[Email]:
    """
    Combines all category email templates into a shuffled list of Email objects.
    Also populates EMAIL_GROUND_TRUTH with labels for each email_id.
    """
    global EMAIL_GROUND_TRUTH

    all_templates = (
        BILLING_EMAILS +
        SUPPORT_EMAILS +
        SPAM_EMAILS +
        SALES_EMAILS +
        HR_EMAILS +
        GENERAL_EMAILS
    )

    senders = [
        "alice@clientcorp.com", "bob@enterprise.io", "noreply@spamworld.net",
        "carol@partnerfirm.com", "dave@bigco.org", "eve@startup.dev",
        "frank@vendor.net", "grace@agency.com", "heidi@customer.co",
        "ivan@prospect.biz",
    ]

    emails = []
    random.shuffle(all_templates)

    for i, template in enumerate(all_templates):
        email_id = f"EMAIL_{i+1:03d}"
        sender = random.choice(senders)
        timestamp = generate_timestamp(days_ago=random.randint(0, 14))

        email = Email(
            email_id=email_id,
            sender=sender,
            subject=template["subject"],
            body=template["body"],
            timestamp=timestamp,
        )
        emails.append(email)

        # Store ground truth
        EMAIL_GROUND_TRUTH[email_id] = {
            "category": template["category"],
            "priority": template["priority"],
            "routing_team": template["routing_team"],
            "reply_keywords": template.get("reply_keywords", []),
        }

    return emails